from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import torch
import torchvision
from torch import Tensor, nn
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchtyping import TensorType
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.nerfacto_field import get_normalized_directions
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.renderers import RGBRenderer
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.models.nerfacto import NerfactoModel

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


@dataclass
class NeuralSemanticFieldConfig(ModelConfig):
    """Config for Neural Semantic field"""

    _target: Type = field(default_factory=lambda: NeuralSemanticFieldModel)

    background_color: Literal["random", "last_sample"] = "last_sample"
    """Whether to randomize the background color."""


class NeuralSemanticFieldModel(Model):

    config: NeuralSemanticFieldConfig

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        return []

    def populate_modules(self):
        # TODO create 3D-Unet here
        # raise NotImplementedError
        self.rgb_loss = MSELoss()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)

        # Query the NeRF model at the ray bundles
        self.rgb_zeros = torch.rand((1, 3), requires_grad=True)
        self.rgb_zeros_param = torch.nn.Parameter(self.rgb_zeros)
        # A fallback model used purely for inference rendering if no other model is specified
        self.fallback_model: Optional[Model] = None

        # Feature extractor
        self.feature_model = FeatureGenerator()

        # Feature Transformer
        self.feature_transformer = UNet()

        # Renderer
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            "feature_network": list(self.feature_model.parameters()),
            "feature_transformer": list(self.feature_transformer.parameters()),
        }

    def get_outputs(self, ray_bundle: RayBundle, batch: Union[Dict[str, Any], None] = None):
        # TODO implement UNET
        # TODO query NeRF
        # TODO do feature conversion + MLP
        # TODO do semantic rendering
        model: Model = self.get_model(batch)

        outs, weights = self.feature_model(ray_bundle, model)

        outs = self.feature_transformer(outs)

        rgb = self.renderer_rgb(rgb=outs[FieldHeadNames.RGB], weights=weights)

        outputs = {"rgb": rgb}

        return outputs

    def forward(self, ray_bundle: RayBundle, batch: Union[Dict[str, Any], None] = None) -> Dict[str, torch.Tensor]:
        return self.get_outputs(ray_bundle, batch)

    def get_metrics_dict(self, outputs, batch: Dict[str, Any]):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        return metrics_dict

    def get_loss_dict(self, outputs, batch: Dict[str, Any], metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
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
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        combined_rgb = torch.cat([image, rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        images_dict = {"img": combined_rgb}

        psnr = self.psnr(image, rgb)
        metrics_dict = {"psnr": float(psnr.item())}

        return metrics_dict, images_dict

    def set_model(self, model: Model):
        """Sets the fallback model for inference of the Neural Semantic Field.
        It is a nerf model which can be queried to obtain points.

        Args:
            model (Model): The fallback nerf model
        """
        self.fallback_model = model

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
        return model


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

        if isinstance(model, NerfactoModel):
            model = cast(NerfactoModel, model)
            if model.collider is not None:
                ray_bundle = model.collider(ray_bundle)

            ray_samples, _, _ = model.proposal_sampler(ray_bundle, density_fns=model.density_fns)
            field_outputs = model.field(ray_samples, compute_normals=model.config.predict_normals)
            weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        else:
            raise NotImplementedError("Only NerfactoModel is supported for now")

        # normalize field densitities:
        densities = field_outputs[FieldHeadNames.DENSITY]
        mean_vals = torch.mean(densities, dim=1, keepdim=True)
        std_vals = torch.std(densities, dim=1, keepdim=True)

        # Scale the values to have a mean of 0 and a standard deviation of 1
        normalized_densities = (densities - mean_vals) / std_vals

        field_outputs_stacked = torch.cat((field_outputs[FieldHeadNames.RGB], normalized_densities), dim=-1)
        field_features = self.mlp_field_output(field_outputs_stacked.view(-1, 4))

        # Positional encoding of the Frustums
        positions_frustums = ray_samples.frustums.get_positions()
        positions_frustums_flat = self.position_frustums_encoding(positions_frustums.view(-1, 3))

        # Positional encoding of the ray
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        pos_encode = torch.cat([d, positions_frustums_flat], dim=1)

        pos_features = self.mlp_merged_pos_encoding(pos_encode)

        features = torch.cat([pos_features, field_features], dim=1)
        features = features.view(ray_samples.shape[0], ray_samples.shape[1], -1)
        return features, weights


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
        return self.double_conv(x)


class Encoder(nn.Module):
    def __init__(self, chs=(1, 64, 128, 256)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(256, 128, 64, 1)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, enc_chs=(1, 16, 32, 64), dec_chs=(64, 32, 16), num_class=4):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        # reduce the channel dimension
        out = torch.mean(out, dim=-1)

        # output in the correct format
        output = {}
        output[FieldHeadNames.RGB] = out[:, :3, :].squeeze(1).permute(0, 2, 1)
        output[FieldHeadNames.DENSITY] = out[:, 3:, :].permute(0, 2, 1)
        return output
