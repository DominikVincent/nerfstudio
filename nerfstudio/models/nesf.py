from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import lovely_tensors as lt
import torch
import torchvision
from rich.console import Console
from torch import Tensor, nn
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchtyping import TensorType
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.dataparsers.base_dataparser import Semantics
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.nerfacto_field import get_normalized_directions
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.renderers import RGBRenderer, SemanticRenderer
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.models.nerfacto import NerfactoModel
from nerfstudio.utils import profiler

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass

lt.monkey_patch()

CONSOLE = Console(width=120)


@dataclass
class NeuralSemanticFieldConfig(ModelConfig):
    """Config for Neural Semantic field"""

    _target: Type = field(default_factory=lambda: NeuralSemanticFieldModel)

    background_color: Literal["random", "last_sample"] = "last_sample"
    """Whether to randomize the background color."""


class NeuralSemanticFieldModel(Model):

    config: NeuralSemanticFieldConfig

    def __init__(self, config: NeuralSemanticFieldConfig, metadata: Dict, **kwargs) -> None:
        assert "semantics" in metadata.keys() and isinstance(metadata["semantics"], Semantics)
        self.semantics = metadata["semantics"]
        super().__init__(config=config, **kwargs)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        return []

    def populate_modules(self):
        # TODO create 3D-Unet here
        # raise NotImplementedError

        # Losses
        self.rgb_loss = MSELoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="mean")

        # Metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)

        # Feature extractor
        # self.feature_model = FeatureGenerator()
        self.feature_model = FeatureGeneratorTorch()

        # Feature Transformer
        self.feature_transformer = UNet(num_class=len(self.semantics.classes))

        # Renderer
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_semantics = SemanticRenderer()

        # This model gets used if no model gets passed in the batch, e.g. when using the viewer
        self.fallback_model: Optional[Model] = None

        # count parameters
        total_params = sum(p.numel() for p in self.parameters())
        CONSOLE.print("The number of NeSF parameters is: ", total_params)

        return

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            "feature_network": list(self.feature_model.parameters()),
            "feature_transformer": list(self.feature_transformer.parameters()),
        }

    @profiler.time_function
    def get_outputs(self, ray_bundle: RayBundle, batch: Union[Dict[str, Any], None] = None):
        # TODO implement UNET
        # TODO query NeRF
        # TODO do feature conversion + MLP
        # TODO do semantic rendering
        model: Model = self.get_model(batch)

        outs, weights = self.feature_model(ray_bundle, model)

        # assert outs is not nan or not inf
        assert not torch.isnan(outs).any()
        assert not torch.isinf(outs).any()

        # assert outs is not nan or not inf
        assert not torch.isnan(outs).any()
        assert not torch.isinf(outs).any()

        field_outputs = self.feature_transformer(outs)

        # rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        semantics = self.renderer_semantics(field_outputs[FieldHeadNames.SEMANTICS], weights=weights)

        # semantics colormaps
        semantic_labels = torch.argmax(torch.nn.functional.softmax(semantics, dim=-1), dim=-1)
        semantics_colormap = self.semantics.colors[semantic_labels].to(self.device)

        outputs = {
            "rgb": semantics_colormap,
            "semantics": semantics,
            "semantics_colormap": semantics_colormap,
        }

        return outputs

    def forward(self, ray_bundle: RayBundle, batch: Union[Dict[str, Any], None] = None) -> Dict[str, torch.Tensor]:
        return self.get_outputs(ray_bundle, batch)

    def get_metrics_dict(self, outputs, batch: Dict[str, Any]):
        metrics_dict = {}
        # image = batch["image"].to(self.device)
        # metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        return metrics_dict

    def get_loss_dict(self, outputs, batch: Dict[str, Any], metrics_dict=None):
        # image = batch["image"].to(self.device)

        pred = outputs["semantics"]
        gt = batch["semantics"][..., 0].long()
        # print the unique values of the gt
        loss_dict = {
            "semantics_loss": self.cross_entropy_loss(pred, gt),
            # "rgb_loss": self.rgb_loss(image, outputs["rgb"])
        }

        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(
        self, camera_ray_bundle: RayBundle, batch: Union[Dict[str, Any], None] = None
    ) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
            :param batch: additional information of the batch here it includes at least the model
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle, batch=batch)
            for output_name, output in outputs.items():  # type: ignore
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            if not torch.is_tensor(outputs_list[0]):
                # TODO: handle lists of tensors as well
                continue
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        semantics_colormap_gt = self.semantics.colors[batch["semantics"].squeeze(-1)].to(self.device)
        semantics_colormap = outputs["semantics_colormap"]
        combined_semantics = torch.cat([semantics_colormap_gt, semantics_colormap], dim=1)

        images_dict = {
            "img": combined_semantics,
            "semantics_colormap": outputs["semantics_colormap"],
        }

        # metrics_dict = {"psnr": float(psnr.item())}
        metrics_dict = {}

        return metrics_dict, images_dict

    def set_model(self, model: Model):
        """Sets the fallback model for inference of the Neural Semantic Field.
        It is a nerf model which can be queried to obtain points.

        Args:
            model (Model): The fallback nerf model
        """
        self.fallback_model = model

    @profiler.time_function
    def get_model(self, batch: Union[Dict[str, Any], None]) -> Model:
        """Gets the fallback model for inference of the Neural Semantic Field.
        It is a nerf model which can be queried to obtain points.

        Returns:
            Model: The fallback nerf model
        """
        if batch is None or "model" not in batch:
            assert self.fallback_model is not None
            model = self.fallback_model
        else:
            model = batch["model"]
        model.eval()
        return model


class FeatureGeneratorTorch(nn.Module):
    """Takes in a batch of b Ray bundles, samples s points along the ray. Then it outputs n x m x f matrix.
    Each row corresponds to one feature of a sampled point of the ray.

    Args:
        nn (_type_): _description_
    """

    def __init__(self, positional_encoding_dim=128, field_output_encoding=128):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.Sigmoid(),
        )
        # self.useless_parameter = nn.Parameter(torch.rand(1, 3))

    def forward(self, ray_bundle: RayBundle, model: Model):
        model.eval()
        if isinstance(model, NerfactoModel):
            model = cast(NerfactoModel, model)
            with torch.no_grad():
                if model.collider is not None:
                    ray_bundle = model.collider(ray_bundle)

                ray_samples, _, _ = model.proposal_sampler(ray_bundle, density_fns=model.density_fns)
                field_outputs = model.field(ray_samples, compute_normals=model.config.predict_normals)
                weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        else:
            raise NotImplementedError("Only NerfactoModel is supported for now")

        rgb = field_outputs[FieldHeadNames.RGB]
        density = field_outputs[FieldHeadNames.DENSITY]

        features = self.linear(rgb.view(-1, 3))
        features = features.view(ray_samples.shape[0], ray_samples.shape[1], -1)
        # add_term = +0.000001 * torch.sigmoid(self.useless_parameter.repeat(rgb.shape[0], rgb.shape[1], 1))
        # features = rgb + add_term
        out = features
        return out, weights


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        assert x.dtype == torch.float32
        y = self.double_conv(x)
        assert y.dtype == torch.float32
        assert not torch.isnan(y).any()

        return y


class Encoder(nn.Module):
    def __init__(self, chs=(1, 64, 128, 256)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x) -> List[torch.Tensor]:
        ftrs = []
        assert x.dtype == torch.float32
        for block in self.enc_blocks:
            x = block(x)
            assert x.dtype == torch.float32

            ftrs.append(x)
            x = self.pool(x)
            assert x.dtype == torch.float32

        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(256, 128, 64, 1)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x: torch.Tensor, encoder_features: List[torch.Tensor]):
        assert x.dtype == torch.float32

        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            assert x.dtype == torch.float32

            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            assert x.dtype == torch.float32

            x = self.dec_blocks[i](x)
            assert x.dtype == torch.float32

        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        assert enc_ftrs.dtype == torch.float32
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, enc_chs=(1, 16, 32), dec_chs=(32, 16), num_class=4):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.density_activation = nn.ReLU()
        self.num_classes = num_class

    def forward(self, x):
        x = x.unsqueeze(1)

        enc_ftrs = self.encoder(x)

        for enc_ftr in enc_ftrs:
            assert not torch.isnan(enc_ftr).any()
            assert not torch.isnan(enc_ftr).any()
            assert enc_ftr.dtype == torch.float32

        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])

        assert not torch.isnan(out).any()
        assert not torch.isnan(out).any()

        out = self.head(out)

        assert not torch.isnan(out).any()
        assert not torch.isnan(out).any()

        # reduce the channel dimension
        out = torch.mean(out, dim=-1)

        output = {}
        output[FieldHeadNames.SEMANTICS] = out.squeeze(1).permute(0, 2, 1)

        # assert that output does not contain nan and inf values
        assert not torch.isnan(output[FieldHeadNames.SEMANTICS]).any()
        assert not torch.isinf(output[FieldHeadNames.SEMANTICS]).any()
        return output


class FeatureGenerator(nn.Module):
    """Takes in a batch of b Ray bundles, samples s points along the ray. Then it outputs n x m x f matrix.
    Each row corresponds to one feature of a sampled point of the ray.

    Args:
        nn (_type_): _description_
    """

    def __init__(self, positional_encoding_dim=128, field_output_encoding=128):
        super().__init__()
        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.position_frustums_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={"otype": "Frequency", "n_frequencies": 2},
        )

        self.mlp_merged_pos_encoding = tcnn.Network(
            n_input_dims=self.direction_encoding.n_output_dims + self.position_frustums_encoding.n_output_dims,
            n_output_dims=positional_encoding_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 4,
            },
        )

        # Tiny cudnn network for processing the field outputs of the samples b x s x (rgb + density = 4)
        self.mlp_field_output = tcnn.Network(
            n_input_dims=4,
            n_output_dims=field_output_encoding,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 4,
            },
        )

    def forward(self, ray_bundle: RayBundle, model: Model) -> Tuple[Tensor, TensorType[..., "num_samples", 1]]:
        model.eval()
        if isinstance(model, NerfactoModel):
            model = cast(NerfactoModel, model)
            with torch.no_grad():
                if model.collider is not None:
                    ray_bundle = model.collider(ray_bundle)

                ray_samples, _, _ = model.proposal_sampler(ray_bundle, density_fns=model.density_fns)
                field_outputs = model.field(ray_samples, compute_normals=model.config.predict_normals)
                weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        else:
            raise NotImplementedError("Only NerfactoModel is supported for now")

        # normalize field densitities:
        densities = field_outputs[FieldHeadNames.DENSITY]
        # set infinite values to highest finite value
        densities[torch.isinf(densities)] = torch.max(densities[~torch.isinf(densities)])
        densities = torch.log(densities + 1.0)

        mean_vals = torch.mean(densities, dim=1, keepdim=True)
        std_vals = torch.std(densities, dim=1, keepdim=True)

        # Scale the values to have a mean of 0 and a standard deviation of 1
        normalized_densities = (densities - mean_vals) / std_vals

        field_outputs_stacked = torch.cat((field_outputs[FieldHeadNames.RGB], normalized_densities), dim=-1)
        field_features = self.mlp_field_output(field_outputs_stacked.view(-1, 4))

        # assert that field_features are not nan or inf
        assert torch.isnan(field_features).sum() == 0
        assert torch.isinf(field_features).sum() == 0

        # Positional encoding of the Frustums
        positions_frustums = ray_samples.frustums.get_positions()
        positions_frustums_flat = self.position_frustums_encoding(positions_frustums.view(-1, 3))

        # Positional encoding of the ray
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        pos_encode = torch.cat([d, positions_frustums_flat], dim=1)

        pos_features = self.mlp_merged_pos_encoding(pos_encode)

        # assert that pos_features are not nan or inf
        assert torch.isnan(pos_features).sum() == 0
        assert torch.isinf(pos_features).sum() == 0

        features = torch.cat([pos_features, field_features], dim=1)
        features = features.view(ray_samples.shape[0], ray_samples.shape[1], -1)
        return features, weights
