from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import torch
from rich.console import Console
from torch.nn import Parameter
from torch.utils.data import DataLoader
from typing_extensions import Literal

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    AnnotatedDataParserUnion,
    DataManager,
)
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.nesf_dataparser import NerfstudioDataParserConfig, Nesf
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.datasets.nesf_dataset import NesfDataset, NesfItemDataset
from nerfstudio.data.pixel_samplers import EquirectangularPixelSampler, PixelSampler
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils import profiler

CONSOLE = Console(width=120)
MAX_AUTO_RESOLUTION = 1600


@dataclass
class NesfDataManagerConfig(InstantiateConfig):
    """Configuration for data manager instantiation; DataManager is in charge of keeping the train/eval dataparsers;
    After instantiation, data manager holds both train/eval datasets and is in charge of returning unpacked
    train/eval data at each iteration
    """

    _target: Type = field(default_factory=lambda: NesfDataManager)
    """Target class to instantiate."""
    dataparser: AnnotatedDataParserUnion = NerfstudioDataParserConfig()
    """Specifies the dataparser used to unpack the data."""
    train_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per training iteration."""
    train_num_images_to_sample_from: int = 4
    """Number of images to sample during training iteration."""
    train_num_times_to_repeat_images: int = 4
    """When not training on all images, number of iterations before picking new
    images. If -1, never pick new images."""
    eval_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per eval iteration."""
    eval_num_images_to_sample_from: int = 4
    """Number of images to sample during eval iteration."""
    eval_num_times_to_repeat_images: int = 4
    """When not evaluating on all images, number of iterations before picking
    new images. If -1, never pick new images."""
    eval_image_indices: Optional[Tuple[int, ...]] = (0,)
    """Specifies the image indices to use during eval; if None, uses all."""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig()
    """Specifies the camera pose optimizer used during training. Helpful if poses are noisy, such as for data from
    Record3D."""
    collate_fn = staticmethod(nerfstudio_collate)
    """Specifies the collate function to use for the train and eval dataloaders."""
    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """
    steps_per_model: int = 6
    """Number of steps one model is queried before the next model is queried. The models are taken sequentially."""


class NesfDataManager(DataManager):  # pylint: disable=abstract-method
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: NesfDataManagerConfig
    train_datasets: NesfDataset
    eval_datasets: NesfDataset
    train_dataparser_outputs: DataparserOutputs
    train_pixel_sampler: Optional[PixelSampler] = None
    eval_pixel_sampler: Optional[PixelSampler] = None

    def __init__(
        self,
        config: NesfDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser: Nesf = self.config.dataparser.setup()
        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(split="train")

        self.train_datasets = self.create_train_datasets()
        self.eval_datasets = self.create_eval_datasets()
        self.train_dataset = self.train_datasets
        self.eval_dataset = self.eval_datasets
        self.eval_image_model = 0
        self.eval_model = 0
        super().__init__()

    def create_train_datasets(self) -> NesfDataset:
        """Sets up the data loaders for training"""
        return NesfDataset(
            [
                NesfItemDataset(dataparser_outputs=dataparser_output, scale_factor=self.config.camera_res_scale_factor)
                for dataparser_output in self.train_dataparser_outputs
            ]
        )

    def create_eval_datasets(self) -> NesfDataset:
        """Sets up the data loaders for evaluation"""
        return NesfDataset(
            [
                NesfItemDataset(dataparser_outputs=dataparser_output, scale_factor=self.config.camera_res_scale_factor)
                for dataparser_output in self.dataparser.get_dataparser_outputs(split=self.test_split)
            ]
        )

    def _get_pixel_sampler(  # pylint: disable=no-self-use
        self, dataset: NesfItemDataset, *args: Any, **kwargs: Any
    ) -> PixelSampler:
        """Infer pixel sampler to use."""
        # If all images are equirectangular, use equirectangular pixel sampler
        is_equirectangular = dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value
        if is_equirectangular.all():
            return EquirectangularPixelSampler(*args, **kwargs)
        # Otherwise, use the default pixel sampler
        if is_equirectangular.any():
            CONSOLE.print("[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler.")
        return PixelSampler(*args, **kwargs)

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_datasets is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloaders = [
            CacheDataloader(
                train_dataset,
                num_images_to_sample_from=self.config.train_num_images_to_sample_from,
                num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
                device=self.device,
                num_workers=self.world_size * 4,
                pin_memory=True,
                collate_fn=self.config.collate_fn,
            )
            for train_dataset in self.train_datasets
        ]
        self.iter_train_image_dataloaders = [
            iter(train_image_dataloader) for train_image_dataloader in self.train_image_dataloaders
        ]

        self.train_pixel_samplers = [
            self._get_pixel_sampler(train_dataset, self.config.train_num_rays_per_batch)
            for train_dataset in self.train_datasets
        ]

        def get_camera_conf(group_name) -> CameraOptimizerConfig:
            self.config.camera_optimizer.param_group = group_name
            return deepcopy(self.config.camera_optimizer)

        self.train_camera_optimizers = [
            get_camera_conf(group_name=get_dir_of_path(dataparser_output.image_filenames[0])).setup(
                num_cameras=train_dataset.cameras.shape[0], device=self.device
            )
            for dataparser_output, train_dataset in zip(self.train_dataparser_outputs, self.train_datasets)
        ]

        self.train_ray_generators = [
            RayGenerator(
                train_dataset.cameras.to(self.device),
                train_camera_optimizer,
            )
            for train_dataset, train_camera_optimizer in zip(self.train_datasets, self.train_camera_optimizers)
        ]

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_datasets is not None
        CONSOLE.print("Setting up evaluation dataset...")
        self.eval_image_dataloaders = [
            CacheDataloader(
                eval_dataset,
                num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
                num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
                device=self.device,
                num_workers=self.world_size * 4,
                pin_memory=True,
                collate_fn=self.config.collate_fn,
            )
            for eval_dataset in self.eval_datasets
        ]

        self.iter_eval_image_dataloaders = [
            iter(eval_image_dataloader) for eval_image_dataloader in self.eval_image_dataloaders
        ]

        print("iters created")
        self.eval_pixel_samplers = [
            self._get_pixel_sampler(eval_dataset, self.config.eval_num_rays_per_batch)
            for eval_dataset in self.eval_datasets
        ]
        self.eval_ray_generators = [
            RayGenerator(
                eval_dataset.cameras.to(self.device),
                train_camera_optimizer,  # should be shared between train and eval.
            )
            for eval_dataset, train_camera_optimizer in zip(self.eval_datasets, self.train_camera_optimizers)
        ]

        # for loading full images
        self.fixed_indices_eval_dataloaders = [
            FixedIndicesEvalDataloader(
                input_dataset=eval_dataset,
                device=self.device,
                num_workers=self.world_size * 4,
            )
            for eval_dataset in self.eval_datasets
        ]

        self.eval_dataloaders = [
            RandIndicesEvalDataloader(
                input_dataset=eval_dataset,
                image_indices=self.config.eval_image_indices,
                device=self.device,
                num_workers=self.world_size * 4,
            )
            for eval_dataset in self.eval_datasets
        ]

    def debug_stats(self):
        non_gpu = []
        for i, dataset in enumerate(self.train_datasets):
            dataset = cast(NesfItemDataset, dataset)
            if dataset.model.device.type != "cpu":
                non_gpu.append(i)
        print("Non gpu: ", non_gpu)

    def models_to_cpu(self, step):
        """Moves all models who shouldnt be active to cpu."""
        model_idx = self.step_to_dataset(step)
        for i, dataset in enumerate(self.train_datasets):
            if i == model_idx:
                continue
            dataset = cast(NesfItemDataset, dataset)
            dataset.model.to("cpu")

    @profiler.time_function
    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        model_idx = self.step_to_dataset(step)
        CONSOLE.print(f"Train model scene {model_idx}")
        image_batch = next(self.iter_train_image_dataloaders[model_idx])
        assert self.train_pixel_samplers[model_idx] is not None
        batch = self.train_pixel_samplers[model_idx].sample(image_batch)
        ray_indices = batch["indices"]
        batch["model_idx"] = model_idx
        batch["model"] = image_batch["model"][0]
        ray_bundle = self.train_ray_generators[model_idx](ray_indices)
        assert str(batch["image"].device) == "cpu"
        assert str(batch["semantics"].device) == "cpu"
        assert str(batch["indices"].device) == "cpu"
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        model_idx = self.eval_model % self.eval_datasets.set_count()
        CONSOLE.print(f"Eval model scene {model_idx}")
        self.eval_model += 1
        image_batch = next(self.iter_eval_image_dataloaders[model_idx])
        assert self.eval_pixel_samplers[model_idx] is not None
        batch = self.eval_pixel_samplers[model_idx].sample(image_batch)
        ray_indices = batch["indices"]
        batch["model_idx"] = model_idx
        batch["model"] = image_batch["model"][0]
        ray_bundle = self.eval_ray_generators[model_idx](ray_indices)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        model_idx = self.eval_image_model % self.eval_datasets.set_count()
        self.eval_image_model += 1

        for camera_ray_bundle, batch in self.eval_dataloaders[model_idx]:
            assert camera_ray_bundle.camera_indices is not None
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])
            return image_idx, model_idx, camera_ray_bundle, batch
        raise ValueError("No more eval images")

    def get_param_groups(self) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        # TODO consider whether this is needed as the models parameters are assumed to be fixed. Potentially return {}
        param_groups = {}
        for train_camera_optimizer in self.train_camera_optimizers:
            camera_opt_params = list(train_camera_optimizer.parameters())
            if train_camera_optimizer.config.mode != "off":
                assert len(camera_opt_params) > 0
                param_groups[self.config.camera_optimizer.param_group] = camera_opt_params
            else:
                assert len(camera_opt_params) == 0

        return param_groups

    def step_to_dataset(self, step: int) -> int:
        """Returns the dataset index for the given step."""
        return (step // self.config.steps_per_model) % self.train_datasets.set_count()

    def steps_to_next_dataset(self, step: int) -> int:
        """Returns the number of steps until the next dataset is used."""
        return self.config.steps_per_model - (step % self.config.steps_per_model)


def get_dir_of_path(path: Path) -> str:
    return str(path.parent.name)
