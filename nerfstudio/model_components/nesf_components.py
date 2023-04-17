import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Tuple, Type, Union, cast

import torch
from pointnet.models.pointnet2_sem_seg import get_model
from rich.console import Console
from torch import nn
from torch.nn import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.encodings import RFFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.nerfacto_field import get_normalized_directions
from nerfstudio.models.base_model import Model
from nerfstudio.models.nerfacto import NerfactoModel
from nerfstudio.utils.nesf_utils import (
    log_points_to_wandb,
    visualize_point_batch,
    visualize_points,
)

CONSOLE = Console(width=120)


@dataclass
class TranformerEncoderModelConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: TransformerEncoderModel)

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


class TransformerEncoderModel(torch.nn.Module):
    def __init__(
        self,
        config: TranformerEncoderModelConfig,
        input_size: int,
        activation: Union[Callable, None] = None,
        pretrain: bool = False,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        self.config = config
        self.input_size = input_size
        self.activation = activation
        self.pretrain = pretrain
        self.mask_ratio = mask_ratio

        # Feature dim layer
        self.feature_dim_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, self.config.feature_transformer_feature_dim),
            torch.nn.ReLU(),
        )

        # Define the transformer encoder
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            self.config.feature_transformer_feature_dim,
            self.config.feature_transformer_num_heads,
            self.config.feature_transformer_dim_feed_forward,
            self.config.feature_transformer_dropout_rate,
            batch_first=True,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            self.encoder_layer, self.config.feature_transformer_num_layers
        )

        if self.pretrain:
            self.mask_token = torch.nn.Parameter(torch.randn(1, 1, input_size), requires_grad=True)
            torch.nn.init.normal_(self.mask_token, std=0.02)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Taken from: https://github.com/facebookresearch/mae/blob/main/models_mae.py
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, batch: dict):
        """
        batch = {
            ids_restore=None
            src_key_padding_mask=None
        }

        If pretrain == False:
            the input x is a sequence of shape [N, L, D] will simply be transformed by the transformer encoder.
            it returns x - where x is the encoded input sequence of shape [N, L, feature_dim]

        If pretrain == True && ids_restore is None:
            then it assumes it is the encoder. The input will be masked and the ids_reorder will be returned together with the transformed input and the mask.
            return x, masks, ids_restore

        If pretrain == True && ids_restore is not None:
            then it assumes it is the decoder. The input will be the masked input and the ids_reorder will be used to reorder the input.
            it retruns x - where x is the encoded input sequence of shape [N, L, feature_dim]
        """

        ids_restore = batch.get("ids_restore", None)
        src_key_padding_mask = batch.get("src_key_padding_mask", None)

        encode = ids_restore is None
        # if encode:
        #     CONSOLE.print("Encoding", style="bold")
        # else:
        #     CONSOLE.print("Decoding", style="bold")

        mask = None
        if self.pretrain and encode:
            x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        elif self.pretrain and not encode:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
            x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
            x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # CONSOLE.print(f"Input shape: {x.shape}")
        x = self.feature_dim_layer(x)  #

        # Apply the transformer encoder. Last step is layer normalization
        x = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )  # {1, num_dense_samples, feature_dim}

        if self.activation is not None:
            x = self.activation(x)

        if self.pretrain and encode:
            return x, mask, ids_restore

        return x

    def get_out_dim(self) -> int:
        return self.config.feature_transformer_feature_dim


@dataclass
class FeatureGeneratorTorchConfig(InstantiateConfig):
    _target: type = field(default_factory=lambda: FeatureGeneratorTorch)

    use_rgb: bool = True
    """Should the rgb be used as a feature?"""
    out_rgb_dim: int = 16
    """The output dimension of the rgb feature"""

    use_density: bool = False
    """Should the density be used as a feature?"""
    out_density_dim: int = 8

    use_pos_encoding: bool = True
    """Should the position encoding be used as a feature?"""

    use_dir_encoding: bool = True
    """Should the direction encoding be used as a feature?"""

    rot_augmentation: bool = True
    """Should the random rot augmentation around the z axis be used?"""
    
    visualize_point_batch: bool = True
    """Visualize the points of the batch? Useful for debugging"""
    
    log_point_batch: bool = False
    """Log the pointcloud to wandb? Useful for debugging. Happens in chance 1/5000"""


class FeatureGeneratorTorch(nn.Module):
    """Takes in a batch of b Ray bundles, samples s points along the ray. Then it outputs n x m x f matrix.
    Each row corresponds to one feature of a sampled point of the ray.

    Args:
        nn (_type_): _description_
    """

    def __init__(self, config: FeatureGeneratorTorchConfig, aabb: TensorType[2, 3]):
        super().__init__()

        self.config: FeatureGeneratorTorchConfig = config

        self.aabb = Parameter(aabb, requires_grad=False)
        self.aabb = cast(TensorType[2, 3], self.aabb)

        if self.config.use_rgb:
            self.rgb_linear = nn.Sequential(
                nn.Linear(3, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, self.config.out_rgb_dim),
            )

        if self.config.use_density:
            self.density_linear = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, self.config.out_density_dim),
            )

        if self.config.use_pos_encoding:
            self.pos_encoder = RFFEncoding(in_dim=3, num_frequencies=8, scale=10)
        if self.config.use_dir_encoding:
            self.dir_encoder = SHEncoding()

    def forward(self, field_outputs: dict, transform_batch: dict):
        """
        Takes a ray bundle filters out non dense points and returns a feature matrix of shape [num_dense_samples, feature_dim]
        used_samples be 1 if surface sampling is enabled ow used_samples = num_ray_samples

        Input:
        - ray_bundle: RayBundle [N]

        Output:
         - out: [1, points, feature_dim]
         - weights: [N, num_ray_samples, 1]
         - density_mask: [N, used_samples]
         - misc:
            - rgb: [N, used_samples, 3]
            - density: [N, used_samples, 1]
            - ray_samples: [N, used_samples]
        """
        device = transform_batch["points_xyz"].device

        encodings = []

        if self.config.use_rgb:
            rgb = field_outputs[FieldHeadNames.RGB]
            assert not torch.isnan(rgb).any()
            assert not torch.isinf(rgb).any()

            rgb_out = self.rgb_linear(rgb)

            assert not torch.isnan(rgb_out).any()
            assert not torch.isinf(rgb_out).any()

            encodings.append(rgb_out)

        if self.config.use_density:
            density = field_outputs[FieldHeadNames.DENSITY]
            # normalize density between 0 and 1
            density = (density - density.min()) / (density.max() - density.min())
            # assert no nan and no inf values
            # assert not torch.isnan(density).any()
            # assert not torch.isinf(density).any()
            if torch.isnan(density).any():
                CONSOLE.print("density has nan values: ", torch.isnan(density).sum())
                density[torch.isnan(density)] = 0.0
            if torch.isinf(density).any():
                CONSOLE.print("density has inf values: ", torch.isinf(density).sum())
                density[torch.isinf(density)] = 1000000.0

            assert not torch.isnan(density).any()
            density = self.density_linear(density)
            assert not torch.isnan(density).any()
            encodings.append(density)

        if self.config.rot_augmentation:
            theta = torch.rand(1) * 2 * torch.pi

            # Construct the rotation matrix
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            rot_matrix = torch.tensor(
                [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]],
                device=device,
            )
        else:
            rot_matrix = torch.eye(3, device=device)

        if self.config.use_pos_encoding:
            positions = transform_batch["points_xyz"]

            positions_normalized = SceneBox.get_normalized_positions(positions, self.aabb)

            if self.config.rot_augmentation:
                positions_normalized = torch.matmul(positions_normalized, rot_matrix)

            # normalize positions at 0 mean and 1 std per batch
            mean = torch.mean(positions_normalized, dim=1).unsqueeze(1)
            # std = torch.std(positions_normalized, dim=1).unsqueeze(1)

            positions_normalized = positions_normalized - mean
            if self.config.visualize_point_batch:
                visualize_point_batch(positions_normalized)

            if random.random() < (1 / 1):
                log_points_to_wandb(positions_normalized)
            
            pos_encoding = self.pos_encoder(positions_normalized)
            assert not torch.isnan(pos_encoding).any()
            encodings.append(pos_encoding)

        if self.config.use_dir_encoding:
            directions = transform_batch["directions"]
            directions = get_normalized_directions(directions)
            if self.config.rot_augmentation:
                directions = torch.matmul(directions, rot_matrix)
            dir_encoding = self.dir_encoder(directions)

            assert not torch.isnan(dir_encoding).any()
            encodings.append(dir_encoding)

        out = torch.cat(encodings, dim=-1)
        # out: 1, num_dense, out_dim
        # weights: num_rays, num_samples, 1
        return out

    def get_out_dim(self) -> int:
        total_dim = 0
        total_dim += self.config.out_rgb_dim if self.config.use_rgb else 0
        total_dim += self.config.out_density_dim if self.config.use_density else 0
        total_dim += self.pos_encoder.get_out_dim() if self.config.use_pos_encoding else 0
        total_dim += self.dir_encoder.get_out_dim() if self.config.use_dir_encoding else 0
        return total_dim


@dataclass
class PointNetWrapperConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: PointNetWrapper)

    out_feature_channels: int = 128
    """The number of features the model should output"""


class PointNetWrapper(nn.Module):
    def __init__(
        self,
        config: PointNetWrapperConfig,
        input_size: int,
        activation: Union[Callable, None] = None,
        pretrain: bool = False,
        mask_ratio: float = 0.75,
    ):
        """
        input_size: the true input feature size, i.e. the number of features per point. Internally the points will be prepended with the featuers.
        """
        super().__init__()
        self.config = config
        self.input_size = input_size
        self.activation = activation
        self.pretrain = pretrain
        self.mask_ratio = mask_ratio

        # PointNet takes xyz + features as input
        self.feature_transformer = get_model(num_classes=config.out_feature_channels, in_channels=input_size + 3)
        self.output_size = config.out_feature_channels

    def forward(self, x: torch.Tensor, batch: dict):
        start_time = time.time()
        # prepend points xyz to points features
        x = torch.cat((batch["points_xyz"], x), dim=-1)
        x = x.permute(0, 2, 1)
        x, l4_points = self.feature_transformer(x)
        if self.activation is not None:
            x = self.activation(x)
        CONSOLE.print("PointNetWrapper forward time: ", time.time() - start_time)

        return x

    def get_out_dim(self) -> int:
        return self.output_size


@dataclass
class SceneSamplerConfig(InstantiateConfig):
    """target class to instantiate"""

    _target: Type = field(default_factory=lambda: SceneSampler)

    samples_per_ray: int = 10
    """How many samples per ray to take"""
    surface_sampling: bool = True
    """Sample only the surface or also the volume"""
    density_threshold: float = 0.7
    """The density threshold for which to not use the points for training"""
    filter_points: bool = True
    """Whether to filter out points for training"""
    z_value_threshold: float = -1
    """What is the minimum z value a point has to have"""
    xy_distance_threshold: float = 1
    """The maximal distance a point can have to z axis to be considered"""
    max_points: int = 16384
    """The maximum number of points to use in one scene. If more are available after filtering, they will be randomly sampled"""


class SceneSampler:
    """_summary_ A class which samples a scene given a ray bundle.
    It will filter out points/ray_samples and batch up the scene.
    """

    def __init__(self, config: SceneSamplerConfig):
        self.config = config

    def sample_scene(self, ray_bundle: RayBundle, model: Model) -> Tuple[RaySamples, torch.Tensor, dict, torch.Tensor]:
        """_summary_
        Samples the model for a given ray bundle. Filters and batches points.

        Args:
            ray_bundle (_type_): A ray bundle. Might be from different cameras.
            model (_type_): A nerf model. Currently NerfactoModel is supported..

        Returns:
        - ray_samples (_type_): The ray samples for the scene which should be used.
        - weights (_type_): The weights for the ray samples which are used.
        - field_outputs (_type_): The field outputs for the ray samples which are used.
        - final_mask (_type_): The mask for the ray samples which are used. Is the shape of the original ray_samples.

        Raises:
            NotImplementedError: _description_
        """
        model.eval()
        if isinstance(model, NerfactoModel):
            model = cast(NerfactoModel, model)
            with torch.no_grad():
                if model.collider is not None:
                    ray_bundle = model.collider(ray_bundle)

                model.proposal_sampler.num_nerf_samples_per_ray = self.config.samples_per_ray
                ray_samples, _, _ = model.proposal_sampler(ray_bundle, density_fns=model.density_fns)
                field_outputs = model.field(ray_samples, compute_normals=model.config.predict_normals)
                weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        else:
            raise NotImplementedError("Only NerfactoModel is supported for now")

        if self.config.surface_sampling:
            ray_samples, weights, field_outputs = self.surface_sampling(ray_samples, weights, field_outputs)

        density_mask = self.get_density_mask(field_outputs)

        pos_mask = self.get_pos_mask(ray_samples)

        total_mask = density_mask & pos_mask

        final_mask = self.get_limit_mask(total_mask)

        ray_samples, weights, field_outputs = self.apply_mask(ray_samples, weights, field_outputs, final_mask)

        return ray_samples, weights, field_outputs, final_mask

    def surface_sampling(self, ray_samples, weights, field_outputs):
        cumulative_weights = torch.cumsum(weights[..., 0], dim=-1)  # [..., num_samples]
        split = torch.ones((*weights.shape[:-2], 1), device=weights.device) * 0.5  # [..., 1]
        median_index = torch.searchsorted(cumulative_weights, split, side="left")  # [..., 1]
        median_index = torch.clamp(median_index, 0, ray_samples.shape[-1] - 1)  # [..., 1]

        field_outputs[FieldHeadNames.RGB] = field_outputs[FieldHeadNames.RGB][
            torch.arange(median_index.shape[0]), median_index.squeeze(), ...
        ].unsqueeze(1)
        field_outputs[FieldHeadNames.DENSITY] = field_outputs[FieldHeadNames.DENSITY][
            torch.arange(median_index.shape[0]), median_index.squeeze(), ...
        ].unsqueeze(1)
        weights = weights[torch.arange(median_index.shape[0]), median_index.squeeze(), ...].unsqueeze(1)
        ray_samples = ray_samples[torch.arange(median_index.shape[0]), median_index.squeeze()].unsqueeze(1)

        return ray_samples, weights, field_outputs

    def get_density_mask(self, field_outputs):
        # true for points to keep
        density = field_outputs[FieldHeadNames.DENSITY]  # 64, 48, 1 (rays, samples, 1)
        density_mask = density > self.config.density_threshold  # 64, 48
        return density_mask.squeeze(-1)

    def get_pos_mask(self, ray_samples):
        # true for points to keep
        points = ray_samples.frustums.get_positions()
        points_dense_mask = (points[..., 2] > self.config.z_value_threshold) & (
            torch.norm(points[..., :2], dim=-1) <= self.config.xy_distance_threshold
        )
        return points_dense_mask

    def get_limit_mask(self, mask):
        num_true = int(torch.sum(mask).item())

        # If there are more than k true values, randomly select which ones to keep
        if num_true > self.config.max_points:
            true_indices = torch.nonzero(mask, as_tuple=True)
            num_true_values = true_indices[0].size(0)

            # Randomly select k of the true indices
            selected_indices = torch.randperm(num_true_values)[: self.config.max_points]

            # Create a new mask with only the selected true values
            new_mask = torch.zeros_like(mask)
            new_mask[true_indices[0][selected_indices], true_indices[1][selected_indices]] = 1
            return new_mask
        else:
            return mask

    def apply_mask(self, ray_samples, weights, field_outputs, mask):
        for k, v in field_outputs.items():
            field_outputs[k] = v[mask]

        ray_samples = ray_samples[mask]
        weights = weights[mask]

        # all but density_mask should have the reduced size
        return ray_samples, weights, field_outputs
