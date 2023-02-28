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
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.field_components.encodings import RFFEncoding, SHEncoding
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

    rgb: bool = True
    """whether tor predict the identiy rgb-> rgb instead of rgb -> semantics."""


class NeuralSemanticFieldModel(Model):

    config: NeuralSemanticFieldConfig

    def __init__(self, config: NeuralSemanticFieldConfig, metadata: Dict, **kwargs) -> None:
        assert "semantics" in metadata.keys() and isinstance(metadata["semantics"], Semantics)
        self.semantics: Semantics = metadata["semantics"]
        super().__init__(config=config, **kwargs)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        return []

    def populate_modules(self):
        # TODO create 3D-Unet here
        # raise NotImplementedError

        # Losses
        if self.config.rgb:
            # self.rgb_loss = MSELoss()
            self.rgb_loss = nn.L1Loss()
        else:
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss(
                weight=torch.tensor([1.0, 32.0, 32.0, 32.0, 32.0, 32.0]), reduction="mean"
            )

        # Metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)

        # Feature extractor
        # self.feature_model = FeatureGenerator()
        self.feature_model = FeatureGeneratorTorch(
            aabb=self.scene_box.aabb, out_rgb_dim=3, rgb=True, pos_encoding=False, dir_encoding=False
        )

        # Feature Transformer
        # TODO make them customizable
        output_size = 3 if self.config.rgb else len(self.semantics.classes)
        activation = torch.nn.Sigmoid() if self.config.rgb else None
        self.feature_transformer = TransformerModel(
            output_size=output_size,
            num_layers=2,
            d_model=self.feature_model.get_out_dim(),
            num_heads=3,
            dff=9,
            dropout_rate=0.1,
            activation=activation,
        )

        # print parameter count of models
        print("Feature Generator has", sum(p.numel() for p in self.feature_model.parameters()), "parameters")
        print("Feature Transformer has", sum(p.numel() for p in self.feature_transformer.parameters()), "parameters")

        # The learnable parameter for the semantic class with low density. Should represent the logits.
        learned_value_dim = 3 if self.config.rgb else len(self.semantics.classes)
        self.learned_low_density_value = torch.nn.Parameter(torch.randn(learned_value_dim))
        # self.learned_low_density_value = torch.randn(learned_value_dim).to("cuda:0") - 1000

        # Renderer
        if self.config.rgb:
            self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        else:
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
            "learned_low_density_params": [self.learned_low_density_value],
        }

    @profiler.time_function
    def get_outputs(self, ray_bundle: RayBundle, batch: Union[Dict[str, Any], None] = None):
        # TODO implement UNET
        # TODO query NeRF
        # TODO do feature conversion + MLP
        # TODO do semantic rendering
        model: Model = self.get_model(batch)

        outs, weights, density_mask = self.feature_model(ray_bundle, model)
        CONSOLE.print("dense values: ", (density_mask).sum().item(), "/", density_mask.numel())

        # assert outs is not nan or not inf
        assert not torch.isnan(outs).any()
        assert not torch.isinf(outs).any()

        # assert outs is not nan or not inf
        assert not torch.isnan(outs).any()
        assert not torch.isinf(outs).any()

        field_outputs = self.feature_transformer(outs)
        outputs = {}

        if self.config.rgb:
            # debug rgb
            rgb = torch.empty((*density_mask.shape, 3), device=self.device)
            rgb[density_mask] = field_outputs[FieldHeadNames.SEMANTICS]
            rgb[~density_mask] = torch.nn.functional.sigmoid(self.learned_low_density_value)
            rgb = self.renderer_rgb(rgb, weights=weights)
            outputs["rgb"] = rgb
            CONSOLE.print("RGB value", torch.sigmoid(self.learned_low_density_value).p)
        else:
            semantics = torch.empty((*density_mask.shape, len(self.semantics.classes)), device=self.device)
            semantics[density_mask] = field_outputs[FieldHeadNames.SEMANTICS]
            semantics[~density_mask] = self.learned_low_density_value

            # debug semantics
            low_density = semantics[~density_mask]
            low_density_semantic_labels = torch.argmax(torch.nn.functional.softmax(low_density, dim=-1), dim=-1)

            high_density = semantics[density_mask]
            high_density_semantic_labels = torch.argmax(torch.nn.functional.softmax(high_density, dim=-1), dim=-1)

            semantics = self.renderer_semantics(semantics, weights=weights)
            outputs["semantics"] = semantics

            # semantics colormaps
            semantic_labels = torch.argmax(torch.nn.functional.softmax(semantics, dim=-1), dim=-1)
            semantics_colormap = self.semantics.colors[semantic_labels].to(self.device)
            outputs["semantics_colormap"] = semantics_colormap
            outputs["rgb"] = semantics_colormap

            # print the count of the different labels
            CONSOLE.print("Label counts:", torch.bincount(semantic_labels.flatten()))

        return outputs

    def forward(self, ray_bundle: RayBundle, batch: Union[Dict[str, Any], None] = None) -> Dict[str, torch.Tensor]:
        return self.get_outputs(ray_bundle, batch)

    def get_metrics_dict(self, outputs, batch: Dict[str, Any]):
        metrics_dict = {}
        if self.config.rgb:
            image = batch["image"].to(self.device)
            metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        return metrics_dict

    def get_loss_dict(self, outputs, batch: Dict[str, Any], metrics_dict=None):
        loss_dict = {}
        if self.config.rgb:
            image = batch["image"].to(self.device)
            loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
        else:
            pred = outputs["semantics"]
            gt = batch["semantics"][..., 0].long()
            CONSOLE.print("GT labels:", torch.bincount(gt.flatten()).p)
            CONSOLE.print(
                "Pred labels:", torch.bincount(torch.argmax(torch.nn.functional.softmax(pred, dim=-1), dim=-1)).p
            )
            # print the unique values of the gt
            loss_dict["semantics_loss"] = (self.cross_entropy_loss(pred, gt),)

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
        images_dict = {}
        metrics_dict = {}

        if self.config.rgb:
            image = batch["image"].to(self.device)
            rgb = outputs["rgb"]
            combined_rgb = torch.cat([image, rgb], dim=1)
            images_dict["img"] = combined_rgb

            # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
            image = torch.moveaxis(image, -1, 0)[None, ...]
            rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

            psnr = self.psnr(image, rgb)
            metrics_dict["psnr"] = float(psnr.item())

        else:
            semantics_colormap_gt = self.semantics.colors[batch["semantics"].squeeze(-1)].to(self.device)
            semantics_colormap = outputs["semantics_colormap"]
            combined_semantics = torch.cat([semantics_colormap_gt, semantics_colormap], dim=1)
            images_dict["img"] = combined_semantics
            images_dict["semantics_colormap"] = outputs["semantics_colormap"]

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

    def __init__(
        self,
        aabb,
        density_threshold: float = 0.5,
        out_rgb_dim: int = 8,
        rgb: bool = True,
        pos_encoding: bool = True,
        dir_encoding: bool = True,
    ):
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.density_threshold = density_threshold
        self.rgb = rgb
        self.pos_encoding = pos_encoding
        self.dir_encoding = dir_encoding

        self.out_rgb_dim: int = out_rgb_dim
        self.linear = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_rgb_dim),
            nn.Sigmoid(),
        )

        if self.pos_encoding:
            self.pos_encoder = RFFEncoding(in_dim=3, num_frequencies=8, scale=10)
        if self.dir_encoding:
            self.dir_encoder = SHEncoding()

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

        density = field_outputs[FieldHeadNames.DENSITY]
        density_mask = (density > self.density_threshold).squeeze(-1)

        encodings = []

        if self.rgb:
            rgb = field_outputs[FieldHeadNames.RGB][density_mask]
            # rgb = rgb + self.linear(rgb)
            encodings.append(rgb)

        if self.pos_encoding:
            positions = ray_samples.frustums.get_positions()[density_mask]
            positions_normalized = SceneBox.get_normalized_positions(positions, self.aabb)
            pos_encoding = self.pos_encoder(positions_normalized)
            encodings.append(pos_encoding)

        if self.dir_encoding:
            directions = ray_samples.frustums.directions[density_mask]
            dir_encoding = self.dir_encoder(get_normalized_directions(directions))
            encodings.append(dir_encoding)

        out = torch.cat(encodings, dim=1).unsqueeze(0)
        return out, weights, density_mask

    def get_out_dim(self) -> int:
        total_dim = 0
        total_dim += self.out_rgb_dim if self.rgb else 0
        total_dim += self.pos_encoder.get_out_dim() if self.pos_encoding else 0
        total_dim += self.dir_encoder.get_out_dim() if self.dir_encoding else 0
        return total_dim


class TransformerModel(torch.nn.Module):
    def __init__(self, output_size, num_layers, d_model, num_heads, dff, dropout_rate, activation=None):
        super().__init__()

        # Define the transformer encoder
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers)

        # Define the output layer
        self.final_layer = torch.nn.Sequential(
            torch.nn.Linear(d_model, 128),
            torch.nn.Linear(128, output_size),
        )

        self.activation = activation

    def forward(self, x):

        x = x.squeeze(0)
        # Apply the transformer encoder
        input_data = x
        x = self.transformer_encoder(x)

        # Apply the final layer

        x = self.final_layer(x)

        # x = x + input_data

        if self.activation is not None:
            x = self.activation(x)

        x = x.unsqueeze(0)

        return {FieldHeadNames.SEMANTICS: input_data + x}
