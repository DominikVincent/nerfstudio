import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from math import ceil, sqrt
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import lovely_tensors as lt
import numpy as np
import plotly.graph_objs as go
import torch
import torch.nn.functional as F
from rich.console import Console
from torch import nn
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.classification import MulticlassJaccardIndex
from typing_extensions import Literal

import wandb
from nerfstudio.cameras.rays import RayBundle, stack_ray_bundles
from nerfstudio.data.dataparsers.base_dataparser import Semantics
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.nesf_components import *
from nerfstudio.model_components.renderers import RGBRenderer, SemanticRenderer
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import profiler
from nerfstudio.utils.writer import put_config

lt.monkey_patch()

CONSOLE = Console(width=120)
DEBUG_PLOT_SAMPLES = False
DEBUG_PLOT_WANDB_SAMPLES = True
DEBUG_FILTER_POINTS = True


@dataclass
class NeuralSemanticFieldConfig(ModelConfig):
    """Config for Neural Semantic field"""

    _target: Type = field(default_factory=lambda: NeuralSemanticFieldModel)

    background_color: Literal["random", "last_sample"] = "last_sample"
    """Whether to randomize the background color."""

    mode: Literal["rgb", "semantics", "density"] = "rgb"
    """The mode in which the model is trained. It predicts whatever mode is chosen. Density is only used for pretraining."""

    use_feature_rgb: bool = True
    """whether to use rgb as feature or not."""
    rgb_feature_dim: int = 16
    """the dimension of the rgb feature."""
    use_feature_pos: bool = True
    """whether to use pos as feature or not."""
    use_feature_dir: bool = True
    """whether to use viewing direction as feature or not."""
    use_feature_density: bool = True
    """whether to use the [0-1] normalized density as a feature."""
    density_feature_dim: int = 8
    """the dimension of the density feature."""
    density_threshold: float = 0.7
    """The threshold value for which to filter out samples."""
    surface_sampling: bool = True
    """Whether to only sample one point on a rays, which corresponds to the surface."""
    rot_augmentation: bool = True
    """Whether to use random rotations around z-axis for data augmentation."""

    feature_transformer_num_layers: int = 6
    """The number of encoding layers in the feature transformer."""
    feature_transformer_num_heads: int = 8
    """The number of multihead attention heads in the feature transformer."""
    feature_transformer_dim_feed_forward: int = 64
    """The dimension of the feedforward network model in the feature transformer."""
    feature_transformer_dropout_rate: float = 0.2
    """The dropout rate in the feature transformer."""
    feature_transformer_feature_dim: int = 64
    """The number of layers the transformer scales up the input dimensionality to the sequence dimensionality."""

    # In case of pretraining we use a decoder together with a linear unit as prediction head.
    decoder_feature_transformer_num_layers: int = 2
    """The number of encoding layers in the feature transformer."""
    decoder_feature_transformer_num_heads: int = 2
    """The number of multihead attention heads in the feature transformer."""
    decoder_feature_transformer_dim_feed_forward: int = 64
    """The dimension of the feedforward network model in the feature transformer."""
    decoder_feature_transformer_dropout_rate: float = 0.2
    """The dropout rate in the feature transformer."""
    decoder_feature_transformer_feature_dim: int = 64
    """The number of layers the transformer scales up the input dimensionality to the sequence dimensionality."""

    pretrain: bool = False
    """Flag indicating whether the model is in pretraining mode or not."""
    mask_ratio: float = 0.5
    """The ratio of pixels that are masked out during pretraining."""

    space_partitioning: Literal["row_wise", "random", "evenly"] = "random"
    """How to partition the image space when rendering."""

    density_prediction: Literal["direct", "integration"] = "direct"
    """How to train the density prediction. With the direct nerf density output or throught the integration process"""
    density_cutoff: float = 1e8
    """Large density values might be an issue for training. Hence they can get cut off with this."""

    batching_mode: Literal["sequential", "sliced", "off"] = "sliced"
    """Usually all samples are fed into the transformer at the same time. This could be too much for the model to understand and also too much for VRAM.
    Hence we batch the samples:
     - sequential: we batch the samples by wrapping them sequentially into batches.
     - sliced: Sort points by x coordinate and then slice them into batches.
     - off: no batching is done."""
    batch_size: int = 2048

    samples_per_ray: int = 10
    """When sampling the underlying nerfs. How many samples should be taken per ray."""


def get_wandb_histogram(tensor):
    def create_histogram(tensor):
        hist, bin_edges = np.histogram(tensor.detach().cpu().numpy(), bins=100)
        return hist, bin_edges

    tensor = tensor.flatten()
    hist = create_histogram(tensor)
    return wandb.Histogram(np_histogram=hist)


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
        if self.config.mode == "rgb":
            # self.rgb_loss = MSELoss()
            self.rgb_loss = nn.L1Loss()
        elif self.config.mode == "semantics":
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss(
                weight=torch.tensor([1.0, 32.0, 32.0, 32.0, 32.0, 32.0]), reduction="mean"
            )
        elif self.config.mode == "density":
            self.density_loss = nn.L1Loss()

        # Metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.miou = MulticlassJaccardIndex(num_classes=len(self.semantics.classes))

        # Feature extractor
        # self.feature_model = FeatureGenerator()
        self.feature_model = FeatureGeneratorTorch(
            aabb=self.scene_box.aabb,
            out_rgb_dim=self.config.rgb_feature_dim,
            out_density_dim=self.config.density_feature_dim,
            rgb=self.config.use_feature_rgb,
            pos_encoding=self.config.use_feature_pos,
            dir_encoding=self.config.use_feature_dir,
            density=self.config.use_feature_density,
            density_threshold=self.config.density_threshold,
            rot_augmentation=self.config.rot_augmentation,
            samples_per_ray=self.config.samples_per_ray,
            surface_sampling=self.config.surface_sampling,
        )

        # Feature Transformer
        # TODO make them customizable
        if self.config.mode == "rgb":
            output_size = 3
        elif self.config.mode == "semantics":
            output_size = len(self.semantics.classes)
        elif self.config.mode == "density":
            output_size = 1
        else:
            raise ValueError(f"Unknown mode {self.config.mode}")

        activation = (
            torch.nn.ReLU() if self.config.mode == "rgb" or self.config.mode == "density" else torch.nn.Identity()
        )
        self.feature_transformer = TransformerEncoderModel(
            input_size=self.feature_model.get_out_dim(),
            feature_dim=self.config.feature_transformer_feature_dim,
            num_layers=self.config.feature_transformer_num_layers,
            num_heads=self.config.feature_transformer_num_heads,
            dim_feed_forward=self.config.feature_transformer_dim_feed_forward,
            dropout_rate=self.config.feature_transformer_dropout_rate,
            activation=activation,
            pretrain=self.config.pretrain,
            mask_ratio=self.config.mask_ratio,
        )

        if self.config.pretrain:
            # TODO add transformer decoder
            self.decoder = TransformerEncoderModel(
                input_size=self.feature_transformer.get_out_dim(),
                feature_dim=self.config.decoder_feature_transformer_feature_dim,
                num_layers=self.config.decoder_feature_transformer_num_layers,
                num_heads=self.config.decoder_feature_transformer_num_heads,
                dim_feed_forward=self.config.decoder_feature_transformer_dim_feed_forward,
                dropout_rate=self.config.decoder_feature_transformer_dropout_rate,
                activation=activation,
                pretrain=self.config.pretrain,
                mask_ratio=self.config.mask_ratio,
            )

            self.head = torch.nn.Sequential(
                torch.nn.Linear(self.decoder.get_out_dim(), output_size),
                torch.nn.ReLU(),
            )
        else:
            self.decoder = torch.nn.Identity()

            self.head = torch.nn.Sequential(
                torch.nn.Linear(self.feature_transformer.get_out_dim(), 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, output_size),
                torch.nn.ReLU(),
            )
        self.learned_low_density_value = torch.nn.Parameter(torch.randn(output_size))

        # Renderer
        if self.config.mode == "rgb":
            self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        elif self.config.mode == "semantics":
            self.renderer_semantics = SemanticRenderer()
        elif self.config.mode == "density":
            self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        else:
            raise ValueError(f"Unknown mode {self.config.mode}")

        # This model gets used if no model gets passed in the batch, e.g. when using the viewer
        self.fallback_model: Optional[Model] = None

        # count parameters
        total_params = sum(p.numel() for p in self.parameters())
        put_config(
            "network parameters",
            {
                "feature_generator_parameters": sum(p.numel() for p in self.feature_model.parameters()),
                "feature_transformer_parameters": sum(p.numel() for p in self.feature_transformer.parameters()),
                "decoder_parameters": sum(p.numel() for p in self.decoder.parameters()),
                "head_parameters": sum(p.numel() for p in self.head.parameters()),
                "total_parameters": total_params,
            },
            0,
        )
        CONSOLE.print("Feature Generator has", sum(p.numel() for p in self.feature_model.parameters()), "parameters")
        CONSOLE.print(
            "Feature Transformer has", sum(p.numel() for p in self.feature_transformer.parameters()), "parameters"
        )
        CONSOLE.print("Decoder has", sum(p.numel() for p in self.decoder.parameters()), "parameters")
        CONSOLE.print("Head has", sum(p.numel() for p in self.head.parameters()), "parameters")

        CONSOLE.print("The number of NeSF parameters is: ", total_params)

        return

    def get_param_groups(self) -> Dict[str, List[Parameter]]:

        param_groups = {
            "feature_network": list(self.feature_model.parameters()),
            "feature_transformer": list(self.feature_transformer.parameters()),
            "learned_low_density_params": [self.learned_low_density_value],
            "decoder": list(self.decoder.parameters()),
            "head": list(self.head.parameters()),
        }

        # filter the empty ones
        for key in list(param_groups.keys()):
            if len(param_groups[key]) == 0:
                del param_groups[key]

        return param_groups

    @profiler.time_function
    def get_outputs(self, ray_bundle: RayBundle, batch: Union[Dict[str, Any], None] = None):
        # TODO implement UNET
        # TODO query NeRF
        # TODO do feature conversion + MLP
        # TODO do semantic rendering
        model: Model = self.get_model(batch)

        outs, weights, density_mask, misc = self.feature_model(ray_bundle, model)  # 1, low_dense, 49
        # CONSOLE.print("dense values: ", (density_mask).sum().item(), "/", density_mask.numel())
        # print("Outs shape: ", outs.shape)
        # print("density_mask shape: ", density_mask.shape)
        # print("density_mask shape: ", density_mask.sum().item())
        # points = misc["ray_samples"].frustums.get_positions()[density_mask].cpu().numpy()
        # scatter = go.Scatter3d(
        #     x=points[:, 0],
        #     y=points[:, 1],
        #     z=points[:, 2],
        #     mode="markers",
        #     marker=dict(size=1, opacity=0.8),
        # )

        # # create a layout with axes labels
        # layout = go.Layout(
        #     title=f"RGB points: {points.shape}",
        #     scene=dict(xaxis=dict(title="X"), yaxis=dict(title="Y"), zaxis=dict(title="Z")),
        # )
        # fig = go.Figure(data=[scatter], layout=layout)
        # fig.show()
        # filter points manually
        if DEBUG_FILTER_POINTS:
            Z_FILTER_VALUE = -2.49
            Z_FILTER_VALUE = -1.0
            XY_DISTANCE = 1.0
            points_dense = misc["ray_samples"].frustums.get_positions()[density_mask]
            points_dense_mask = (points_dense[:, 2] > Z_FILTER_VALUE) & (
                torch.norm(points_dense[:, :2], dim=1) <= XY_DISTANCE
            )
            points_dense_mask = points_dense_mask.to(density_mask.device)
            outs = outs[:, points_dense_mask, :]

            points = misc["ray_samples"].frustums.get_positions()
            points_mask = (points[:, :, 2] > Z_FILTER_VALUE) & (torch.norm(points[:, :, :2], dim=2) <= XY_DISTANCE).to(
                density_mask.device
            )
            density_mask = torch.logical_and(density_mask, points_mask)
            MAX_COUNT_POINTS = 16384
            if density_mask.sum() > MAX_COUNT_POINTS:
                CONSOLE.print("There are too many points, we are limiting to ", MAX_COUNT_POINTS, " points.")
                # randomly mask points such that we have MAX_COUNT_POINTS points
                outs = outs[:, :MAX_COUNT_POINTS, :]
                true_indices = torch.nonzero(density_mask.flatten()).squeeze()
                true_indices = true_indices[:MAX_COUNT_POINTS]
                density_mask = torch.zeros_like(density_mask)
                density_mask.view(-1, 1)[true_indices] = 1

        # assert outs is not nan or not inf
        assert not torch.isnan(outs).any()
        assert not torch.isinf(outs).any()

        # assert outs is not nan or not inf
        assert not torch.isnan(outs).any()
        assert not torch.isinf(outs).any()
        if self.config.batching_mode != "off":
            W = outs.shape[1]

            if self.config.batching_mode == "sequential":
                padding = (
                    self.config.batch_size - (W % self.config.batch_size) if W % self.config.batch_size != 0 else 0
                )
                outs = torch.nn.functional.pad(outs, (0, 0, 0, padding))
                masking_top = torch.full((W,), False, dtype=torch.bool)
                masking_bottom = torch.full((padding,), True, dtype=torch.bool)
                masking = torch.cat((masking_top, masking_bottom)).to(self.device)
                ids_shuffle = torch.arange(outs.shape[1])
                ids_restore = ids_shuffle
            elif self.config.batching_mode == "sliced":
                cell_content = self.config.batch_size
                columns_per_row = max(int(sqrt(W / cell_content)), 1)
                row_size = columns_per_row * cell_content
                row_count = ceil(outs.shape[1] / row_size)
                print(
                    "Row count",
                    row_count,
                    "columns per row",
                    columns_per_row,
                    "Grid size",
                    row_count * columns_per_row,
                    "Takes points: ",
                    row_count * columns_per_row * cell_content,
                    "At point count: ",
                    W,
                )
                padding = (
                    (columns_per_row * self.config.batch_size) - (W % (columns_per_row * self.config.batch_size))
                    if W % (columns_per_row * self.config.batch_size) != 0
                    else 0
                )
                outs = torch.nn.functional.pad(outs, (0, 0, 0, padding))
                masking_top = torch.full((W,), False, dtype=torch.bool)
                masking_bottom = torch.full((padding,), True, dtype=torch.bool)
                masking = torch.cat((masking_top, masking_bottom)).to(self.device)

                points = misc["ray_samples"].frustums.get_positions()[density_mask].view(-1, 3)
                rand_points = torch.randn((padding, 3), device=points.device)
                points_padded = torch.cat((points, rand_points))
                ids_shuffle = torch.argsort(points_padded[:, 0])

                for i in range(row_count):
                    # for each row seperate into smaller cells by y axis
                    row_idx = ids_shuffle[i * row_size : (i + 1) * row_size]
                    points_row_y = points_padded[row_idx, 1]
                    ids_shuffle[i * row_size : (i + 1) * row_size] = row_idx[torch.argsort(points_row_y)]

                ids_restore = torch.argsort(ids_shuffle)
                if DEBUG_PLOT_SAMPLES or DEBUG_PLOT_WANDB_SAMPLES:
                    points_pad = torch.nn.functional.pad(points, (0, 0, 0, padding))
                    colors_pad = torch.nn.functional.pad(misc["rgb"][density_mask], (0, 0, 0, padding))
                    for i in range(0, W // self.config.batch_size, 10):
                        break
                        points = (
                            ray_samples.frustums.get_positions()[density_mask].detach().cpu().numpy()[:num_points, :]
                        )
                        colors = field_outputs[FieldHeadNames.RGB][density_mask].detach().cpu().numpy()

                        points = (
                            points_pad[ids_shuffle[self.config.batch_size * i : self.config.batch_size * (i + 1)], :]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        colors = (
                            colors_pad[ids_shuffle[self.config.batch_size * i : self.config.batch_size * (i + 1)], :]
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        scatter = go.Scatter3d(
                            x=points[:, 0],
                            y=points[:, 1],
                            z=points[:, 2],
                            mode="markers",
                            marker=dict(color=colors, size=1, opacity=0.8),
                        )

                        # create a layout with axes labels
                        layout = go.Layout(
                            title=f"RGB points: {points.shape}",
                            scene=dict(xaxis=dict(title="X"), yaxis=dict(title="Y"), zaxis=dict(title="Z")),
                        )
                        fig = go.Figure(data=[scatter], layout=layout)
                        fig.show()
            else:
                raise ValueError(f"Unknown batching mode {self.config.batching_mode}")

            # sorting by ids rearangement. Not necessary for sequential batching
            outs = outs[:, ids_shuffle, :]
            mask = masking[ids_shuffle]

            # batching
            outs = outs.reshape(-1, self.config.batch_size, outs.shape[-1])
            mask = mask.reshape(-1, self.config.batch_size)

            if DEBUG_PLOT_SAMPLES:
                points_pad_reordered = points_pad[ids_shuffle, :].to("cpu")
                points_pad_reordered = points_pad_reordered.reshape(-1, self.config.batch_size, 3)
                points = torch.empty((0, 3))
                colors = torch.empty((0, 3))
                for i in range(points_pad_reordered.shape[0]):
                    points = torch.cat((points, points_pad_reordered[i, :, :]))
                    colors = torch.cat((colors, torch.randn((1, 3)).repeat(self.config.batch_size, 1) * 255))
                scatter = go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                    marker=dict(color=colors, size=1, opacity=0.8),
                )

                # create a layout with axes labels
                layout = go.Layout(
                    title=f"RGB points: {points.shape}",
                    scene=dict(xaxis=dict(title="X"), yaxis=dict(title="Y"), zaxis=dict(title="Z")),
                )
                fig = go.Figure(data=[scatter], layout=layout)
                fig.show()

            if DEBUG_PLOT_WANDB_SAMPLES and random.random() < (1 / 5000):
                points_pad_reordered = points_pad[ids_shuffle, :].to("cpu")
                points_pad_reordered = points_pad_reordered.reshape(-1, self.config.batch_size, 3)
                points = torch.empty((0, 3))
                colors = torch.empty((0, 3))
                for i in range(points_pad_reordered.shape[0]):
                    points = torch.cat((points, points_pad_reordered[i, :, :]))
                    colors = torch.cat((colors, torch.randn((1, 3)).repeat(self.config.batch_size, 1) * 255))
                point_cloud = torch.cat((points, colors), dim=1).detach().cpu().numpy()
                wandb.log({"point_samples": wandb.Object3D(point_cloud)}, step=wandb.run.step)
        else:
            ids_restore = None
            mask = None
            W = None

        outputs = {}
        if self.config.pretrain:
            x, mask, ids_restore = self.feature_transformer(outs, src_key_padding_mask=mask)
            x = self.decoder(x, ids_restore)
            field_outputs = self.head(x)

            # comment to investigate how random masking works
            # take x and replace mask_ratio of its element with random values in [0,1]
            # num_cols_to_mask = int(outs.shape[1] * self.config.mask_ratio)
            # indices = torch.randperm(outs.shape[1])[:num_cols_to_mask]

            # # repeat learned low density value to get the shape of the mask
            # ldv = torch.nn.functional.relu(self.learned_low_density_value)
            # outs[:, indices, :] = 0.000001 * ldv.repeat(outs.shape[0], num_cols_to_mask, 1)
            # # random masking
            # # outs[:, indices, :] = torch.rand(outs.shape[0], num_cols_to_mask, outs.shape[2], device=self.device)
            # # mask with mean value
            # print("ldv: ", misc["rgb"].shape)
            # ldv = torch.mean(misc["rgb"].view(-1, 3), dim=0)
            # print("ldv: ", ldv.shape, " outs: ", outs.shape)
            # outs[:, indices, :] = ldv.repeat(outs.shape[0], num_cols_to_mask, 1)

            # field_outputs = outs
        else:
            field_encodings = self.feature_transformer(outs, src_key_padding_mask=mask)
            field_encodings = self.decoder(field_encodings)
            field_outputs = self.head(field_encodings)

        # unbatch the data
        if self.config.batching_mode != "off":
            field_outputs = field_outputs.reshape(1, -1, field_outputs.shape[-1])
            # reshuffle results
            field_outputs = field_outputs[:, ids_restore, :]

            # removed padding token
            field_outputs = field_outputs[:, :W, :]

        if self.config.mode == "rgb":
            # debug rgb
            rgb = torch.empty((*density_mask.shape, 3), device=self.device)
            rgb[density_mask] = field_outputs
            rgb[~density_mask] = torch.nn.functional.relu(self.learned_low_density_value)

            rgb = self.renderer_rgb(rgb, weights=weights)
            outputs["rgb"] = rgb
        elif self.config.mode == "semantics":
            semantics = torch.empty((*density_mask.shape, len(self.semantics.classes)), device=self.device)  # 64, 48, 6
            print(
                "semantics: ",
                semantics.shape,
                " field_outputs: ",
                field_outputs.shape,
                " density_mask: ",
                density_mask.shape,
                "density_mask max: ",
                density_mask.max(),
                " density_mask: ",
                density_mask,
            )
            semantics[density_mask] = field_outputs  # 1, num_dense_samples, 6
            # semantics[~density_mask] = self.learned_low_density_value
            semantics[~density_mask] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device)

            # debug semantics
            # low_density = semantics[~density_mask]
            # low_density_semantic_labels = torch.argmax(torch.nn.functional.softmax(low_density, dim=-1), dim=-1)

            # high_density = semantics[density_mask]
            # high_density_semantic_labels = torch.argmax(torch.nn.functional.softmax(high_density, dim=-1), dim=-1)

            semantics = self.renderer_semantics(semantics, weights=weights)
            outputs["semantics"] = semantics  # 64, 6

            # semantics colormaps
            semantic_labels = torch.argmax(torch.nn.functional.softmax(semantics, dim=-1), dim=-1)
            self.semantics.colors = self.semantics.colors.to(self.device)
            semantics_colormap = self.semantics.colors[semantic_labels].to(self.device)
            outputs["semantics_colormap"] = semantics_colormap
            outputs["rgb"] = semantics_colormap

            # print the count of the different labels
            # CONSOLE.print("Label counts:", torch.bincount(semantic_labels.flatten()))
        elif self.config.mode == "density":
            rgb = misc["rgb"]
            density = torch.empty((*density_mask.shape, 1), device=self.device)
            density[density_mask] = field_outputs
            density[~density_mask] = torch.nn.functional.relu(self.learned_low_density_value)
            # make it predict logarithmic density instead
            density = torch.exp(density) - 1

            weights = misc["ray_samples"].get_weights(density)

            rgb = self.renderer_rgb(rgb, weights=weights)
            outputs["rgb"] = rgb
            outputs["density_pred"] = density

            # filter out high density values > 4000 and set them to maximum
            density_gt = misc["density"]
            density_gt[density_gt > self.config.density_cutoff] = self.config.density_cutoff
            outputs["density_gt"] = density_gt
            # with p =0.1 log histograms
            if random.random() < 0.1:
                if wandb.run is not None:
                    wandb.log(
                        {
                            "density/pred": get_wandb_histogram(density),
                            "density/gt": get_wandb_histogram(density_gt),
                        },
                        step=wandb.run.step,
                    )
        else:
            raise ValueError("Unknown mode: " + self.config.mode)
        return outputs

    def forward(self, ray_bundle: RayBundle, batch: Union[Dict[str, Any], None] = None) -> Dict[str, torch.Tensor]:
        return self.get_outputs(ray_bundle, batch)

    def enrich_dict_with_model(self, d: dict, model_idx: int) -> dict:
        keys = list(d.keys())

        for key in keys:
            d[key + "_" + str(model_idx)] = d[key]
        d["model_idx"] = model_idx

        return d

    def get_metrics_dict(self, outputs, batch: Dict[str, Any]):
        metrics_dict = {}
        if "eval_model_idx" in batch:
            metrics_dict["eval_model_idx"] = batch["eval_model_idx"]

        if self.config.mode == "rgb":
            with torch.no_grad():
                image = batch["image"].to(self.device)
                metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
                metrics_dict["psnr_" + str(batch["model_idx"])] = metrics_dict["psnr"]

                metrics_dict["mse"] = F.mse_loss(outputs["rgb"], image)
                metrics_dict["mse_" + str(batch["model_idx"])] = metrics_dict["mse"]

                metrics_dict["mae"] = F.l1_loss(outputs["rgb"], image)
                metrics_dict["mae_" + str(batch["model_idx"])] = metrics_dict["mae"]

        elif self.config.mode == "semantics":
            semantics = batch["semantics"][..., 0].long().to(self.device)
            with torch.no_grad():
                # mIoU
                metrics_dict["miou_" + str(batch["model_idx"])] = self.miou(outputs["semantics"], semantics)
        elif self.config.mode == "density":
            image = batch["image"].to(self.device)

            with torch.no_grad():
                metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
                metrics_dict["psnr_" + str(batch["model_idx"])] = metrics_dict["psnr"]

                metrics_dict["mse"] = F.mse_loss(outputs["rgb"], image)
                metrics_dict["mse_" + str(batch["model_idx"])] = metrics_dict["mse"]

                metrics_dict["mae"] = F.l1_loss(outputs["rgb"], image)
                metrics_dict["mae_" + str(batch["model_idx"])] = metrics_dict["mae"]

                metrics_dict["density_mse"] = F.mse_loss(outputs["density_gt"], outputs["density_pred"])
                metrics_dict["density_mse" + str(batch["model_idx"])] = metrics_dict["density_mse"]

                metrics_dict["density_mae"] = F.l1_loss(outputs["density_gt"], outputs["density_pred"])
                metrics_dict["density_mae_" + str(batch["model_idx"])] = metrics_dict["density_mae"]

        return metrics_dict

    def get_loss_dict(self, outputs, batch: Dict[str, Any], metrics_dict=None):
        loss_dict = {}
        if self.config.mode == "rgb":
            image = batch["image"].to(self.device)

            model_output = outputs["rgb"]

            loss_dict["rgb_loss_" + str(batch["model_idx"])] = self.rgb_loss(image, model_output)
        elif self.config.mode == "semantics":
            pred = outputs["semantics"]
            gt = batch["semantics"][..., 0].long().to(self.device)
            loss_dict["semantics_loss_" + str(batch["model_idx"])] = self.cross_entropy_loss(pred, gt)
        elif self.config.mode == "density":
            if self.config.density_prediction == "direct":
                loss_dict["density"] = self.density_loss(outputs["density_gt"], outputs["density_pred"])
            elif self.config.density_prediction == "integration":
                image = batch["image"].to(self.device)
                model_output = outputs["rgb"]
                loss_dict["density"] = self.density_loss(image, model_output)
            else:
                raise ValueError("Unknown density prediction mode: " + self.config.density_prediction)
        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(
        self, camera_ray_bundle: Union[RayBundle, List[RayBundle]], batch: Union[Dict[str, Any], None] = None
    ) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
            :param batch: additional information of the batch here it includes at least the model
        """
        if isinstance(camera_ray_bundle, list):
            images = len(camera_ray_bundle)
            camera_ray_bundle = stack_ray_bundles(camera_ray_bundle)
            image_height, image_width = camera_ray_bundle.origins.shape[:2]
            image_height = image_height // images
            use_all_pixels = False
        else:
            use_all_pixels = True
            image_height, image_width = camera_ray_bundle.origins.shape[:2]

        def batch_randomly(max_length, batch_size):
            indices = torch.randperm(max_length)
            reverse_indices = torch.argsort(indices)
            return indices, reverse_indices

        def batch_evenly(max_length, batch_size):
            indices = torch.arange(max_length)
            final_indices = []
            reverse_indices = torch.zeros(max_length, dtype=torch.long)

            step_size = max_length // batch_size + 1
            running_length = 0
            for i in range(step_size):
                ind = indices[i::step_size]
                length_ind = ind.size()[0]
                final_indices.append(ind)
                reverse_indices[ind] = torch.arange(running_length, running_length + length_ind, dtype=torch.long)
                running_length += length_ind

            return torch.cat(final_indices, dim=0), reverse_indices

        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        if self.config.space_partitioning == "evenly":
            ray_order, reversed_ray_order = batch_evenly(num_rays, num_rays_per_chunk)
        elif self.config.space_partitioning == "random":
            ray_order, reversed_ray_order = batch_randomly(num_rays, num_rays_per_chunk)
        else:
            ray_order = []
            reversed_ray_order = None

        # get permuted ind
        for i in range(0, num_rays, num_rays_per_chunk):
            if self.config.space_partitioning != "row_wise":
                indices = ray_order[i : i + num_rays_per_chunk]
                ray_bundle = camera_ray_bundle.flatten()[indices]
            else:
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
            if self.config.space_partitioning != "row_wise":
                unordered_output_tensor = torch.cat(outputs_list)
                ordered_output_tensor = unordered_output_tensor[reversed_ray_order]
            else:
                ordered_output_tensor = torch.cat(outputs_list)  # type: ignore

            if not use_all_pixels:
                ordered_output_tensor = ordered_output_tensor[: image_height * image_width]
            outputs[output_name] = ordered_output_tensor.view(image_height, image_width, -1)  # type: ignore
        return outputs

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        images_dict = {}
        metrics_dict = {}

        if self.config.mode == "rgb" or self.config.mode == "density":
            image = batch["image"].to(self.device)
            rgb = outputs["rgb"]
            combined_rgb = torch.cat([image, rgb], dim=1)
            images_dict["img"] = combined_rgb

            # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
            image = torch.moveaxis(image, -1, 0)[None, ...]
            rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

            with torch.no_grad():
                psnr = self.psnr(image, rgb)
                metrics_dict["psnr"] = float(psnr.item())
                # metrics_dict["psnr_" + str(batch["model_idx"])] = metrics_dict["psnr"]

                metrics_dict["mse"] = F.mse_loss(image, rgb)
                # metrics_dict["mse_" + str(batch["model_idx"])] = metrics_dict["mse"]

                metrics_dict["mae"] = F.l1_loss(image, rgb)
                # metrics_dict["mae_" + str(batch["model_idx"])] = metrics_dict["mae"]

                metrics_dict["ssim"] = self.ssim(image, rgb)
                # metrics_dict["ssim_" + str(batch["model_idx"])] = metrics_dict["ssim"]

                if self.config.mode == "density":
                    metrics_dict["density_mse"] = F.mse_loss(outputs["density_gt"], outputs["density_pred"])

                    metrics_dict["density_mae"] = F.l1_loss(outputs["density_gt"], outputs["density_pred"])

        elif self.config.mode == "semantics":
            semantics_colormap_gt = self.semantics.colors[batch["semantics"].squeeze(-1)].to(self.device)
            semantics_colormap = outputs["semantics_colormap"]
            combined_semantics = torch.cat([semantics_colormap_gt, semantics_colormap], dim=1)
            images_dict["img"] = combined_semantics
            images_dict["semantics_colormap"] = outputs["semantics_colormap"]
            images_dict["rgb_image"] = batch["image"]

            outs = outputs["semantics"].reshape(-1, outputs["semantics"].shape[-1]).to(self.device)
            gt = batch["semantics"][..., 0].long().reshape(-1)
            miou = self.miou(outs, gt)
            metrics_dict = {"miou": float(miou.item())}

        return metrics_dict, images_dict

    def set_model(self, model: Model):
        """Sets the fallback model for inference of the Neural Semantic Field.
        It is a nerf model which can be queried to obtain points.

        Args:
            model (Model): The fallback nerf model
        """
        # set the model to not require a gradient
        for param in model.parameters():
            param.requires_grad = False
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
            CONSOLE.print("Using fallback model for inference")
            model = self.fallback_model
        else:
            # CONSOLE.print("Using batch model for inference")
            model = batch["model"]
        model.eval()
        return model
