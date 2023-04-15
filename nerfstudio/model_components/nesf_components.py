import time
from typing import Callable, Union, cast

import torch
from pointnet.models.pointnet2_sem_seg import get_model
from rich.console import Console
from torch import nn
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.encodings import RFFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.nerfacto_field import get_normalized_directions
from nerfstudio.models.base_model import Model
from nerfstudio.models.nerfacto import NerfactoModel
from nerfstudio.utils.nesf_utils import visualize_points

CONSOLE = Console(width=120)


class TransformerEncoderModel(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        feature_dim: int = 32,
        num_layers: int = 6,
        num_heads: int = 4,
        dim_feed_forward: int = 64,
        dropout_rate: float = 0.1,
        activation: Union[Callable, None] = None,
        pretrain: bool = False,
        mask_ratio: float = 0.75,
    ):
        super().__init__()

        # Feature dim layer
        self.feature_dim = feature_dim
        self.feature_dim_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, feature_dim),
            torch.nn.ReLU(),
        )

        # Define the transformer encoder
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            feature_dim, num_heads, dim_feed_forward, dropout_rate, batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers)

        self.activation = activation

        self.pretrain = pretrain
        self.mask_ratio = mask_ratio
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
        return self.feature_dim


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
        out_density_dim: int = 8,
        rgb: bool = True,
        pos_encoding: bool = True,
        dir_encoding: bool = True,
        density: bool = True,
        rot_augmentation: bool = False,
        samples_per_ray: int = 10,
        surface_sampling: bool = True,
    ):
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.density_threshold = density_threshold
        self.rgb = rgb

        self.pos_encoding = pos_encoding
        self.dir_encoding = dir_encoding
        self.density = density

        self.out_rgb_dim: int = out_rgb_dim
        self.out_density_dim: int = out_density_dim

        self.rot_augmentation: bool = rot_augmentation
        self.samples_per_ray = samples_per_ray
        self.surface_sampling = surface_sampling

        self.rgb_linear = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_rgb_dim),
        )
        self.density_linear = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.out_density_dim),
        )
        if self.pos_encoding:
            self.pos_encoder = RFFEncoding(in_dim=3, num_frequencies=8, scale=10)
        if self.dir_encoding:
            self.dir_encoder = SHEncoding()

    def forward(self, ray_bundle: RayBundle, model: Model):
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
        model.eval()
        if isinstance(model, NerfactoModel):
            model = cast(NerfactoModel, model)
            with torch.no_grad():
                if model.collider is not None:
                    ray_bundle = model.collider(ray_bundle)

                model.proposal_sampler.num_nerf_samples_per_ray = self.samples_per_ray
                ray_samples, _, _ = model.proposal_sampler(ray_bundle, density_fns=model.density_fns)
                field_outputs = model.field(ray_samples, compute_normals=model.config.predict_normals)
                weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

                if self.surface_sampling:
                    cumulative_weights = torch.cumsum(weights[..., 0], dim=-1)  # [..., num_samples]
                    split = torch.ones((*weights.shape[:-2], 1), device=weights.device) * 0.5  # [..., 1]
                    median_index = torch.searchsorted(cumulative_weights, split, side="left")  # [..., 1]
                    # median_index2 = torch.searchsorted(cumulative_weights, split, side="right")  # [..., 1]
                    # median_index = torch.cat((median_index, median_index2), dim=-1)
                    median_index = torch.clamp(median_index, 0, ray_samples.shape[-1] - 1)  # [..., 1]

                    field_outputs[FieldHeadNames.RGB] = field_outputs[FieldHeadNames.RGB][
                        torch.arange(median_index.shape[0]), median_index.squeeze(), ...
                    ].unsqueeze(1)
                    field_outputs[FieldHeadNames.DENSITY] = field_outputs[FieldHeadNames.DENSITY][
                        torch.arange(median_index.shape[0]), median_index.squeeze(), :
                    ].unsqueeze(1)
                    ray_samples = ray_samples[torch.arange(median_index.shape[0]), median_index.squeeze()].unsqueeze(1)
        else:
            raise NotImplementedError("Only NerfactoModel is supported for now")

        misc = {
            "rgb": field_outputs[FieldHeadNames.RGB],  # 64, 48, 4
            "density": field_outputs[FieldHeadNames.DENSITY],  # 64, 48, 1
            "ray_samples": ray_samples,  # 64, 48, ...
        }
        density = field_outputs[FieldHeadNames.DENSITY]  # 64, 48, 1 (rays, samples, 1)
        density_mask = (density > self.density_threshold).squeeze(-1)  # 64, 48

        encodings = []

        if self.rgb:
            rgb = field_outputs[FieldHeadNames.RGB][density_mask]
            assert not torch.isnan(rgb).any()
            assert not torch.isinf(rgb).any()

            rgb_out = self.rgb_linear(rgb)

            assert not torch.isnan(rgb_out).any()
            assert not torch.isinf(rgb_out).any()

            encodings.append(rgb_out)

        if self.density:
            density = field_outputs[FieldHeadNames.DENSITY][density_mask]
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

        if self.rot_augmentation:
            theta = torch.rand(1) * 2 * torch.pi

            # Construct the rotation matrix
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            rot_matrix = torch.tensor(
                [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]], device=model.device
            )
        else:
            rot_matrix = torch.eye(3, device=model.device)

        if self.pos_encoding:
            positions = ray_samples.frustums.get_positions()[density_mask]
            positions_normalized = SceneBox.get_normalized_positions(positions, self.aabb)

            if self.rot_augmentation:
                positions_normalized = torch.matmul(positions_normalized, rot_matrix)

            print(
                "Scene ranges from",
                list(torch.min(positions_normalized, dim=0)[0].cpu().numpy()),
                "to",
                list(torch.max(positions_normalized, dim=0)[0].cpu().numpy()),
            )
            visualize_points(positions_normalized)

            pos_encoding = self.pos_encoder(positions_normalized)
            assert not torch.isnan(pos_encoding).any()
            encodings.append(pos_encoding)

        if self.dir_encoding:
            directions = ray_samples.frustums.directions[density_mask]
            directions = get_normalized_directions(directions)
            if self.rot_augmentation:
                directions = torch.matmul(directions, rot_matrix)
            dir_encoding = self.dir_encoder(directions)

            assert not torch.isnan(dir_encoding).any()
            encodings.append(dir_encoding)

        out = torch.cat(encodings, dim=1).unsqueeze(0)
        # out: 1, num_dense, out_dim
        # weights: num_rays, num_samples, 1
        return out, weights, density_mask, misc

    def get_out_dim(self) -> int:
        total_dim = 0
        total_dim += self.out_rgb_dim if self.rgb else 0
        total_dim += self.pos_encoder.get_out_dim() if self.pos_encoding else 0
        total_dim += self.dir_encoder.get_out_dim() if self.dir_encoding else 0
        total_dim += self.out_density_dim if self.density else 0
        return total_dim


class PointNetWrapper(nn.Module):
    def __init__(self, output_channels, in_feature_channels):
        super().__init__()
        # PointNet takes xyz + features as input
        self.feature_transformer = get_model(num_classes=output_channels, in_channels=in_feature_channels + 3)
        self.output_size = output_channels

    def forward(self, x: torch.Tensor, batch: dict):
        start_time = time.time()
        # prepend points xyz to points features
        x = torch.cat((batch["points_xyz"], x), dim=-1)
        x = x.permute(0, 2, 1)
        x, l4_points = self.feature_transformer(x)
        print("PointNetWrapper forward time: ", time.time() - start_time)
        return x

    def get_out_dim(self) -> int:
        return self.output_size
