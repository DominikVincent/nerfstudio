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
        self.rgb_loss = MSELoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="mean")

        # Metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)

        # Feature extractor
        # self.feature_model = FeatureGenerator()
        self.feature_model = FeatureGeneratorTorch(aabb=self.scene_box.aabb)

        # Feature Transformer
        # TODO make them customizable
        self.feature_transformer = TransformerModel(
            output_size=len(self.semantics.classes),
            num_layers=2,
            d_model=self.feature_model.get_out_dim(),
            num_heads=4,
            dff=64,
            dropout_rate=0.1,
        )

        # The learnable parameter for the semantic class with low density. Should represent the logits.
        self.class_probs_low_density = torch.nn.Parameter(torch.randn(len(self.semantics.classes)))

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

        outs, weights, density_mask = self.feature_model(ray_bundle, model)

        # assert outs is not nan or not inf
        assert not torch.isnan(outs).any()
        assert not torch.isinf(outs).any()

        # assert outs is not nan or not inf
        assert not torch.isnan(outs).any()
        assert not torch.isinf(outs).any()

        field_outputs = self.feature_transformer(outs)

        semantics = torch.empty((*density_mask.shape, len(self.semantics.classes)), device=self.device)
        semantics[density_mask] = field_outputs[FieldHeadNames.SEMANTICS]
        semantics[~density_mask] = self.class_probs_low_density
        semantics = self.renderer_semantics(semantics, weights=weights)

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

    def __init__(self, aabb, density_threshold: float = 0.5, out_rgb_dim: int = 8):
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.density_threshold = density_threshold

        self.out_rgb_dim: int = out_rgb_dim
        self.linear = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_rgb_dim),
            nn.Sigmoid(),
        )

        self.pos_encoder = RFFEncoding(in_dim=3, num_frequencies=8, scale=10)
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

        rgb = field_outputs[FieldHeadNames.RGB][density_mask]
        rgb = self.linear(rgb)

        positions = ray_samples.frustums.get_positions()[density_mask]
        positions_normalized = SceneBox.get_normalized_positions(positions, self.aabb)
        pos_encoding = self.pos_encoder(positions_normalized)

        directions = ray_samples.frustums.directions[density_mask]
        dir_encoding = self.dir_encoder(get_normalized_directions(directions))

        out = torch.cat([rgb, pos_encoding, dir_encoding], dim=1).unsqueeze(0)
        return out, weights, density_mask

    def get_out_dim(self) -> int:
        return self.pos_encoder.get_out_dim() + self.dir_encoder.get_out_dim() + self.out_rgb_dim


class TransformerModel(torch.nn.Module):
    def __init__(self, output_size, num_layers, d_model, num_heads, dff, dropout_rate):
        super().__init__()

        # Define the transformer encoder
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers)

        # Define the output layer
        self.final_layer = torch.nn.Linear(d_model, output_size)

    def forward(self, x):
        # Apply the transformer encoder
        x = self.transformer_encoder(x)

        # Apply the final layer
        x = self.final_layer(x)

        return {FieldHeadNames.SEMANTICS: x}
