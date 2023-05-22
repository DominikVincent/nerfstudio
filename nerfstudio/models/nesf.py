import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from math import ceil, sqrt
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import lovely_tensors as lt
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.graph_objs as go
import torch
import torch.nn.functional as F
import tyro
from rich.console import Console
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.classification import ConfusionMatrix
from typing_extensions import Literal

import wandb
from nerfstudio.cameras.rays import RayBundle, stack_ray_bundles
from nerfstudio.data.dataparsers.base_dataparser import Semantics
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import pred_normal_loss
from nerfstudio.model_components.nesf_components import *
from nerfstudio.model_components.renderers import (
    NormalsRenderer,
    RGBRenderer,
    SemanticRenderer,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import profiler
from nerfstudio.utils.nesf_utils import *
from nerfstudio.utils.writer import put_config

lt.monkey_patch()

CONSOLE = Console(width=120)


@dataclass
class NeuralSemanticFieldConfig(ModelConfig):
    """Config for Neural Semantic field"""

    _target: Type = field(default_factory=lambda: NeuralSemanticFieldModel)

    background_color: Literal["random", "last_sample"] = "last_sample"
    """Whether to randomize the background color."""

    mode: Literal["rgb", "semantics", "density", "normals"] = "rgb"
    """The mode in which the model is trained. It predicts whatever mode is chosen. Density is only used for pretraining."""

    sampler: SceneSamplerConfig = SceneSamplerConfig()
    """The sampler used in the model."""

    feature_generator_config: FeatureGeneratorTorchConfig = FeatureGeneratorTorchConfig()
    """The feature generating model to use."""

    # feature_transformer_config: AnnotatedTransformerUnion = PointNetWrapperConfig()
    # dirty workaround because Union of configs didnt work
    feature_transformer_model: Literal["pointnet", "custom", "stratified"] = "custom"
    feature_transformer_pointnet_config: PointNetWrapperConfig = PointNetWrapperConfig()
    feature_transformer_custom_config: TranformerEncoderModelConfig = TranformerEncoderModelConfig()
    feature_transformer_stratified_config: StratifiedTransformerWrapperConfig = StratifiedTransformerWrapperConfig()

    # In case of pretraining we use a decoder together with a linear unit as prediction head.
    feature_decoder_model: Literal["pointnet", "custom", "stratified"] = "custom"
    feature_decoder_pointnet_config: PointNetWrapperConfig = PointNetWrapperConfig()
    feature_decoder_custom_config: TranformerEncoderModelConfig = TranformerEncoderModelConfig()
    feature_decoder_stratified_config: StratifiedTransformerWrapperConfig = StratifiedTransformerWrapperConfig()
    """If pretraining is used, what should the encoder look like"""

    masker_config: MaskerConfig = MaskerConfig()
    """If pretraining is used the masker will be used mask poi"""

    pretrain: bool = False
    """Flag indicating whether the model is in pretraining mode or not."""

    space_partitioning: Literal["row_wise", "random", "evenly"] = "random"
    """How to partition the image space when rendering."""

    density_prediction: Literal["direct", "integration"] = "direct"
    """How to train the density prediction. With the direct nerf density output or throught the integration process"""
    density_cutoff: float = 1e8
    """Large density values might be an issue for training. Hence they can get cut off with this."""
    
    rgb_prediction: Literal["direct", "integration"] = "integration"
    """How to train the rgb prediction. With the direct nerf density output or throught the integration process"""

    batching_mode: Literal["sequential", "random", "sliced", "off"] = "off"
    """Usually all samples are fed into the transformer at the same time. This could be too much for the model to understand and also too much for VRAM.
    Hence we batch the samples:
     - sequential: we batch the samples by wrapping them sequentially into batches.
     - random: take random permuatations of points for batching.
     - sliced: Sort points by x coordinate and then slice them into batches.
     - off: no batching is done."""
    batch_size: int = 1536

    debug_show_image: bool = False
    """Show the generated image."""
    debug_show_final_point_cloud: bool = True
    """Show the generated image."""
    log_confusion_to_wandb: bool = True
    plot_confusion: bool = False


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
            self.rgb_loss = torch.nn.L1Loss()
        elif self.config.mode == "semantics":
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss(
                # weight=torch.tensor([1.0, 32.0, 32.0, 32.0, 32.0, 32.0]),
                reduction="mean"
            )
        elif self.config.mode == "density":
            self.density_loss = torch.nn.L1Loss()

        # Metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.confusion = ConfusionMatrix(task="multiclass", num_classes=len(self.semantics.classes), normalize="true")
        self.confusion_non_normalized = ConfusionMatrix(task="multiclass", num_classes=len(self.semantics.classes), normalize="none")

        self.scene_sampler: SceneSampler = self.config.sampler.setup()

        # Feature extractor
        self.feature_model: FeatureGeneratorTorch = self.config.feature_generator_config.setup(aabb=self.scene_box.aabb)

        # Feature Transformer
        # TODO make them customizable
        if self.config.mode == "rgb":
            output_size = 3
        elif self.config.mode == "semantics":
            output_size = len(self.semantics.classes)
        elif self.config.mode == "density":
            output_size = 1
        elif self.config.mode == "normals":
            output_size = 3
        else:
            raise ValueError(f"Unknown mode {self.config.mode}")

        # restrict to positive values if predicting density or rgb
        activation = (
            torch.nn.ReLU() if self.config.mode == "rgb" or self.config.mode == "density" else torch.nn.Identity()
        )
        if self.config.feature_transformer_model == "pointnet":
            self.feature_transformer = self.config.feature_transformer_pointnet_config.setup(
                input_size=self.feature_model.get_out_dim(),
                activation=activation,
            )
        elif self.config.feature_transformer_model == "custom":
            self.feature_transformer = self.config.feature_transformer_custom_config.setup(
                input_size=self.feature_model.get_out_dim(),
                activation=activation,
            )
        elif self.config.feature_transformer_model == "stratified":
            self.feature_transformer = self.config.feature_transformer_stratified_config.setup(
                input_size=self.feature_model.get_out_dim(),
                activation=activation,
            )
        else:
            raise ValueError(f"Unknown feature transformer config {self.config.feature_transformer_model}")

        if self.config.pretrain:
            self.masker: Masker = self.config.masker_config.setup(output_size=self.feature_transformer.get_out_dim())

        if self.config.pretrain:
            # TODO add transformer decoder
            if self.config.feature_decoder_model == "pointnet":
                self.decoder = self.config.feature_decoder_pointnet_config.setup(
                    input_size=self.feature_transformer.get_out_dim(),
                    activation=activation,
                )
            elif self.config.feature_decoder_model == "custom":
                self.decoder = self.config.feature_decoder_custom_config.setup(
                    input_size=self.feature_transformer.get_out_dim(),
                    activation=activation,
                )
            elif self.config.feature_decoder_model == "stratified":
                self.decoder = self.config.feature_decoder_stratified_config.setup(
                    input_size=self.feature_transformer.get_out_dim(),
                    activation=activation,
                )
            else:
                raise ValueError(f"Unknown feature transformer config {self.config.feature_decoder_model}")

            self.head = torch.nn.Sequential(
                torch.nn.Linear(self.decoder.get_out_dim(), output_size),
                activation,
            )
        else:
            self.decoder = torch.nn.Identity()

            self.head = torch.nn.Sequential(
                torch.nn.Linear(self.feature_transformer.get_out_dim(), 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, output_size),
                activation,
            )
        self.learned_low_density_value = torch.nn.Parameter(torch.randn(output_size) * 0.1 + 0.8)

        # Renderer
        if self.config.mode == "rgb":
            self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        elif self.config.mode == "semantics":
            self.renderer_semantics = SemanticRenderer()
        elif self.config.mode == "density":
            self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        elif self.config.mode == "normals":
            # No rendering if we predict normals
            self.renderer_normals = NormalsRenderer()
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
        masker_param_list = list(self.masker.parameters()) if self.config.pretrain else []
        param_groups = {
            "feature_network": list(self.feature_model.parameters()),
            "feature_transformer": list(self.feature_transformer.parameters()),
            "learned_low_density_params": [self.learned_low_density_value] + masker_param_list,
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
        def get_fake_output(size: int):
            # gt_value = torch.zeros(size, self.learned_low_density_value.shape[0]).to(self.learned_low_density_value.device)
            pred_value = self.learned_low_density_value.repeat(size, 1)
            outputs = {
                "rgb": pred_value,
                "density": pred_value,
                "semantics": pred_value,
            }
            return outputs
        outputs = get_fake_output(ray_bundle.shape[0])
        return outputs
        
        model: Model = self.get_model(batch)

        # all but density mask are by filtered dimension
        time1 = time.time()
        (
            ray_samples,
            weights,
            field_outputs_raw,
            density_mask,
            original_fields_outputs,
        ) = self.scene_sampler.sample_scene(ray_bundle, model, batch["model_idx"])
        all_samples_count = density_mask.shape[0]*density_mask.shape[1]
        CONSOLE.print("Ray samples used", weights.shape[0], "out of", all_samples_count, "samples")
        
        time2 = time.time()
        # potentially batch up and infuse field outputs with random points
        field_outputs_raw, transform_batch = self.batching(ray_samples, field_outputs_raw)
        time3 = time.time()
        # TODO return the transformed points
        outs, transform_batch = self.feature_model(field_outputs_raw, transform_batch)  # 1, low_dense, 49
        # assert outs is not nan or not inf
        assert not torch.isnan(outs).any()
        assert not torch.isinf(outs).any()

        # assert outs is not nan or not inf
        assert not torch.isnan(outs).any()
        assert not torch.isinf(outs).any()
        time4 = time.time()
        
        CONSOLE.print("sampling: ", time2 - time1)
        CONSOLE.print("batching: ", time3 - time2)
        CONSOLE.print("feature model: ", time4 - time3)
        CONSOLE.print("Processing pointcloud: ", outs.shape)
        outputs = {}
        if self.config.pretrain:
            # remove all the masked points from outs. The masks tells which points got removed.
            outs, mask, ids_restore, batch = self.masker.mask(outs, transform_batch)

            outs = self.feature_transformer(outs, batch=transform_batch)

            outs, transform_batch = self.masker.unmask(outs, transform_batch, ids_restore)

            outs = self.decoder(outs, batch=transform_batch)

            field_outputs = self.head(outs)
            
            # mask_tokens = torch.tensor([[[0,0,0]]]).repeat(field_outputs.shape[0], ids_restore.shape[1] - field_outputs.shape[1], 1).to(self.device)
            # x_ = torch.cat([field_outputs, mask_tokens], dim=1)  # no cls token
            # field_outputs = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, field_outputs.shape[2]))  # unshuffle

            # TODO maybe use this to only make loss go through the masked points
            # outputs["mask"] = mask

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
            # print("outs: ", outs.shape)
            field_encodings = self.feature_transformer(outs, batch=transform_batch)
            time5 = time.time()
            field_encodings = self.decoder(field_encodings)
            time6 = time.time()
            field_outputs = self.head(field_encodings)
            time7 = time.time()
            CONSOLE.print("feature transformer: ", time5 - time4)
            CONSOLE.print("decoder: ", time6 - time5)
            CONSOLE.print("head: ", time7 - time6)

        # unbatch the data
        if self.config.batching_mode != "off":
            field_outputs = field_outputs.reshape(1, -1, field_outputs.shape[-1])
            for k, v in field_outputs_raw.items():
                field_outputs_raw[k] = v.reshape(1, -1, v.shape[-1])

            # reshuffle results
            field_outputs = field_outputs[:, transform_batch["ids_restore"], :]
            for k, v in field_outputs_raw.items():
                field_outputs_raw[k] = v[:, transform_batch["ids_restore"], :]
            
            # removed padding token
            field_outputs = field_outputs[:, : ray_samples.shape[0], :]
            for k, v in field_outputs_raw.items():
                field_outputs_raw[k] = v[:, : ray_samples.shape[0], :]

        if self.config.mode == "rgb":
            # debug rgb
            rgb = torch.empty((*density_mask.shape, 3), device=self.device)
            weights_all = torch.zeros((*density_mask.shape, 1), device=self.device)  # 64, 48, 6

            rgb[density_mask] = field_outputs
            weights_all[density_mask] = weights

            rgb[~density_mask] = torch.nn.functional.relu(self.learned_low_density_value)
            weights_all[~density_mask] = 0.01

            rgb_integrated = self.renderer_rgb(rgb, weights=weights_all)
            outputs["rgb"] = rgb_integrated
            rgb_gt = original_fields_outputs[FieldHeadNames.RGB]
            outputs["rgb_gt"] = rgb_gt
            outputs["rgb_pred"] = rgb
        elif self.config.mode == "semantics":
            semantics = torch.zeros((*density_mask.shape, len(self.semantics.classes)), device=self.device)  # 64, 48, 6
            weights_all = torch.zeros((*density_mask.shape, 1), device=self.device)  # 64, 48, 6

            # print(semantics.numel() * semantics.element_size() / 1024**2)

            # print(
            #     "semantics: ",
            #     semantics.shape,
            #     " field_outputs: ",
            #     field_outputs.shape,
            #     " density_mask: ",
            #     density_mask.shape,
            #     "density_mask sum: ",
            #     density_mask.sum(),
            #     " density_mask: ",
            #     density_mask,
            #     " field_outputs: ",
            #     field_outputs.shape,
            # )
            semantics[density_mask] = field_outputs  # 1, num_dense_samples, 6
            weights_all[density_mask] = weights

            # semantics[~density_mask] = self.learned_low_density_value
            semantics[~density_mask] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device)
            weights_all[~density_mask] = 0.01

            semantics = self.renderer_semantics(semantics, weights=weights_all)
            outputs["semantics"] = semantics  # N, num_classes

            # semantics colormaps
            semantic_labels = torch.argmax(torch.nn.functional.softmax(semantics, dim=-1), dim=-1)

            self.semantics.colors = self.semantics.colors.to(self.device)
            semantics_colormap = self.semantics.colors[semantic_labels].to(self.device)
            outputs["semantics_colormap"] = semantics_colormap
            outputs["rgb"] = semantics_colormap

            # print the count of the different labels
            # CONSOLE.print("Label counts:", torch.bincount(semantic_labels.flatten()))
        elif self.config.mode == "density":
            density = torch.empty((*density_mask.shape, 1), device=self.device)
            density[density_mask] = field_outputs
            density[~density_mask] = torch.nn.functional.relu(self.learned_low_density_value)
            # make it predict logarithmic density instead
            density = torch.exp(density) - 1

            weights = original_fields_outputs["ray_samples"].get_weights(density)

            rgb = self.renderer_rgb(original_fields_outputs[FieldHeadNames.RGB], weights=weights)
            outputs["rgb"] = rgb
            outputs["density_pred"] = density

            # filter out high density values > 4000 and set them to maximum
            density_gt = original_fields_outputs[FieldHeadNames.DENSITY]
            density_gt[density_gt > self.config.density_cutoff] = self.config.density_cutoff
            outputs["density_gt"] = density_gt
            # with p =0.1 log histograms
            if random.random() < 0.01:
                if wandb.run is not None:
                    wandb.log(
                        {
                            "density/pred": get_wandb_histogram(density),
                            "density/gt": get_wandb_histogram(density_gt),
                        },
                        step=wandb.run.step,
                    )
        elif self.config.mode == "normals":
            
                
            time7=time.time()
            # normalize to unit vecctors
            field_outputs = field_outputs / torch.norm(field_outputs, dim=-1, keepdim=True)
            
            outputs["normals_pred"] = field_outputs
            outputs["normals_gt"] = field_outputs_raw[FieldHeadNames.PRED_NORMALS] if FieldHeadNames.PRED_NORMALS in field_outputs_raw else field_outputs_raw[FieldHeadNames.NORMALS] 
            time8 = time.time()
            
            # for visualization
            normals_all = torch.empty((*density_mask.shape, 3), device=self.device)
            weights_all = torch.zeros((*density_mask.shape, 1), device=self.device)  # 64, 48, 6

            normals_all[density_mask] = field_outputs
            weights_all[density_mask] = weights

            normals_all[~density_mask] = torch.nn.functional.relu(self.learned_low_density_value)
            weights_all[~density_mask] = 0.00001
            
            time9 = time.time()
            
            original_normals = original_fields_outputs[FieldHeadNames.PRED_NORMALS] if FieldHeadNames.PRED_NORMALS in original_fields_outputs else original_fields_outputs[FieldHeadNames.NORMALS]
            outputs["normals_all_pred"] = self.renderer_normals(normals_all, weights=weights_all)
            outputs["normals_all_gt"] = self.renderer_normals(original_normals, weights=weights_all)
            time10 = time.time()
            
            print("saving normals from fieldoutputs", time8-time7)
            print("Creating weights and normals all", time9-time8)
            print("Rendering normals all", time10-time9)
        else:
            raise ValueError("Unknown mode: " + self.config.mode)
        
        outputs["density_mask"] = density_mask
        torch.cuda.empty_cache()

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
                confusion = self.confusion_non_normalized(torch.argmax(outputs["semantics"], dim=-1), semantics).detach().cpu().numpy()
                miou, per_class_iou = compute_mIoU(confusion)
                # mIoU
                metrics_dict["miou"] = miou
                for i, iou in enumerate(per_class_iou):
                    metrics_dict["iou_" + self.semantics.classes[i]] = iou

                metrics_dict["miou_" + str(batch["model_idx"])] = metrics_dict["miou"]
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
        elif self.config.mode == "normals":
            
            metrics_dict["dot"] = (1.0 - torch.sum(outputs["normals_gt"] * outputs["normals_pred"], dim=-1)).mean(dim=-1)
            metrics_dict["dot_" + str(batch["model_idx"])] = metrics_dict["dot"]
            
        return metrics_dict

    def get_loss_dict(self, outputs, batch: Dict[str, Any], metrics_dict=None):
        loss_dict = {}
        if self.config.mode == "rgb":
            image = batch["image"].to(self.device)

            model_output = outputs["rgb"]
            if self.config.rgb_prediction == "integration":
                loss_dict["rgb_loss_" + str(batch["model_idx"])] = self.rgb_loss(image, model_output)
            elif self.config.rgb_prediction == "direct":
                loss_dict["rgb_loss_" + str(batch["model_idx"])] = self.rgb_loss(outputs["rgb_pred"], outputs["rgb_gt"])
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
        elif self.config.mode == "normals":
            loss_dict["dot"] = metrics_dict["dot"]
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
                if  output_name == "normals_pred" or output_name == "normals_gt":
                    continue
                # TODO maybe fix later
                    unordered_output_tensor = torch.cat(outputs_list, dim=1).squeeze(0)
                else:
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

                metrics_dict["mse"] = F.mse_loss(image, rgb).item()
                # metrics_dict["mse_" + str(batch["model_idx"])] = metrics_dict["mse"]

                metrics_dict["mae"] = F.l1_loss(image, rgb).item()
                # metrics_dict["mae_" + str(batch["model_idx"])] = metrics_dict["mae"]

                metrics_dict["ssim"] = self.ssim(image, rgb).item()
                # metrics_dict["ssim_" + str(batch["model_idx"])] = metrics_dict["ssim"]

                if self.config.mode == "density":
                    metrics_dict["density_mse"] = F.mse_loss(outputs["density_gt"], outputs["density_pred"]).item()

                    metrics_dict["density_mae"] = F.l1_loss(outputs["density_gt"], outputs["density_pred"]).item()

        elif self.config.mode == "semantics":
            semantics_colormap_gt = self.semantics.colors[batch["semantics"].squeeze(-1).to("cpu")].to(self.device)
            semantics_colormap = outputs["semantics_colormap"]
            combined_semantics = torch.cat([semantics_colormap_gt, semantics_colormap], dim=1)
            images_dict["img"] = combined_semantics
            images_dict["semantics_colormap"] = outputs["semantics_colormap"]
            images_dict["rgb_image"] = batch["image"]
            
            # compute uncertainty entropy
            def compute_entropy(tensor):
                # Reshape tensor to [B, 256*256, C]
                tensor = tensor.view(tensor.size(0), -1, tensor.size(-1))
                
                # Compute softmax probabilities
                probabilities = F.softmax(tensor, dim=-1)
                
                # Compute entropy
                entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-10), dim=-1)
                
                # Normalize entropy to [0, 1]
                max_entropy = torch.log2(torch.tensor(tensor.size(-1)).float())
                normalized_entropy = entropy / max_entropy
                
                return normalized_entropy.unsqueeze(-1)

            images_dict["entropy"] = compute_entropy(outputs["semantics"])

            outs = outputs["semantics"].reshape(-1, outputs["semantics"].shape[-1]).to(self.device)
            gt = batch["semantics"][..., 0].long().reshape(-1)
            confusion_non_normalized = self.confusion_non_normalized(torch.argmax(outs, dim=-1), gt).detach().cpu().numpy()
            miou, per_class_iou = compute_mIoU(confusion_non_normalized)
            # mIoU
            metrics_dict["miou"] = miou
            metrics_dict["confusion_unnormalized"] = confusion_non_normalized
            for i, iou in enumerate(per_class_iou):
                metrics_dict["iou_" + self.semantics.classes[i]] = iou
                

            confusion = self.confusion(torch.argmax(outs, dim=-1), gt).detach().cpu().numpy()
            if self.config.plot_confusion:
                fig = ff.create_annotated_heatmap(confusion, x=self.semantics.classes, y=self.semantics.classes)
                fig.update_layout(title="Confusion matrix", xaxis_title="Predicted", yaxis_title="Actual")
                fig.show()

            if self.config.log_confusion_to_wandb and wandb.run is not None:
                fig = ff.create_annotated_heatmap(confusion, x=self.semantics.classes, y=self.semantics.classes)
                fig.update_layout(title="Confusion matrix", xaxis_title="Predicted", yaxis_title="Actual")
                wandb.log(
                    {"confusion_matrix": fig},
                    step=wandb.run.step,
                )
        elif self.config.mode == "normals":
            if "normals_all_pred" in outputs:
                images_dict["normals_all_pred"] = (outputs["normals_all_pred"] + 1.0) / 2.0
            if "normals_all_gt" in outputs:
                images_dict["normals_all_gt"] = (outputs["normals_all_gt"] + 1.0) / 2.0
        # plotly show image
        if self.config.debug_show_image:
            fig = px.imshow(images_dict["img"].cpu().numpy())
            fig.show()
        
        if "density_mask" in outputs:
            images_dict["sample_mask"] = torch.mean(outputs["density_mask"].float(), dim=-1, keepdim=True)

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

    def batching(self, ray_samples: RaySamples, field_outputs: dict):
        if self.config.batching_mode != "off":
            if self.config.batching_mode == "sequential":
                # given the features and the density mask batch them up sequentially such that each batch is same size.
                field_outputs, masking, ids_shuffle, ids_restore, points_pad, directions_pad = sequential_batching(
                    ray_samples, field_outputs, self.config.batch_size
                )
            elif self.config.batching_mode == "random":
                field_outputs, masking, ids_shuffle, ids_restore, points_pad, directions_pad = random_batching(
                    ray_samples, field_outputs, self.config.batch_size
                )
            elif self.config.batching_mode == "sliced":
                field_outputs, masking, ids_shuffle, ids_restore, points_pad, directions_pad = spatial_sliced_batching(
                    ray_samples, field_outputs, self.config.batch_size, self.scene_box.aabb
                )
            else:
                raise ValueError(f"Unknown batching mode {self.config.batching_mode}")

            # sorting by ids rearangement. Not necessary for sequential batching
            for k, v in field_outputs.items():
                field_outputs[k] = v[ids_shuffle, :]

            points_pad = points_pad[ids_shuffle, :]
            directions_pad = directions_pad[ids_shuffle, :]
            mask = masking[ids_shuffle]

            # batching
            for k, v in field_outputs.items():
                field_outputs[k] = v.reshape(-1, self.config.batch_size, v.shape[-1])
            mask = mask.reshape(-1, self.config.batch_size)

            transform_batch = {
                "ids_shuffle": ids_shuffle,
                "ids_restore": ids_restore,
                "src_key_padding_mask": mask,
                "points_xyz": points_pad.reshape(*mask.shape, 3),
                "directions": directions_pad.reshape(*mask.shape, 3),
            }
        else:
            W = None
            transform_batch = {
                "ids_shuffle": None,
                "ids_restore": None,
                "src_key_padding_mask": None,
                "points_xyz": ray_samples.frustums.get_positions().unsqueeze(0),
                "directions": ray_samples.frustums.directions.unsqueeze(0),
            }

            for k, v in field_outputs.items():
                field_outputs[k] = v.unsqueeze(0)

        return field_outputs, transform_batch
