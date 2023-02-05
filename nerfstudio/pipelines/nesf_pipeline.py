from __future__ import annotations

import typing
from dataclasses import dataclass, field
from inspect import Parameter
from time import time
from typing import Any, Dict, Optional, Type

import torch
import torch.distributed as dist
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from typing_extensions import Literal

from nerfstudio.configs import base_config as cfg
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManagerConfig,
)
from nerfstudio.data.datamanagers.nesf_datamanager import NesfDataManager
from nerfstudio.engine.callbacks import TrainingCallbackAttributes, TrainingCallback
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, Pipeline
from nerfstudio.utils import profiler


@dataclass
class NesfPipelineConfig(cfg.InstantiateConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: NesfPipeline)
    """target class to instantiate"""
    datamanager: NesfDataManager = NesfDataManager()
    """specifies the datamanager config"""
    model: ModelConfig = ModelConfig()
    """specifies the model config"""


class NesfPipeline(Pipeline):
    """The pipeline class for the nesf nerf setup of multiple cameras for one or a few scenes.

            config: configuration to instantiate pipeline
            device: location to place model and data
            test_mode:
                'val': loads train/val datasets into memory
                'test': loads train/test dataset into memory
                'inference': does not load any dataset into memory
            world_size: total number of machines available
            local_rank: rank of current machine

        Attributes:
            datamanager: The data manager that will be used
            model: The model that will be used
        """

    def __init__(
            self,
            config: NesfPipelineConfig,
            device: str,
            test_mode: Literal["test", "val", "inference"] = "val",
            world_size: int = 1,
            local_rank: int = 0,
    ):
        super().__init__()
        self.config = config
        self.test_mode = test_mode

        self.datamanager: NesfDataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)

        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch = self.datamanager.next_train(step)
        transformer_model_outputs = self.model(ray_bundle, batch)

        metrics_dict = self.model.get_metrics_dict(transformer_model_outputs, batch)

        # No need for camera opt param groups as the nerfs are assumed to be fixed already.

        loss_dict = self.model.get_loss_dict(transformer_model_outputs, batch, metrics_dict)

        return transformer_model_outputs, loss_dict, metrics_dict

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        transformer_model_outputs = self.model(ray_bundle, batch)
        metrics_dict = self.model.get_metrics_dict(transformer_model_outputs, batch)
        loss_dict = self.model.get_loss_dict(transformer_model_outputs, batch, metrics_dict)
        self.train()
        return transformer_model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, batch)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = image_idx
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = len(camera_ray_bundle)
        self.train()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_eval_image_metrics(self, step: Optional[int] = None):
        """Iterate over all the images in the eval dataset and get the average.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        num_images = sum( [len(fixed_indices_eval_dataloader) for fixed_indices_eval_dataloader in self.datamanager.fixed_indices_eval_dataloaders])
        with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                MofNCompleteColumn(),
                transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for fixed_indices_eval_dataloader in self.datamanager.fixed_indices_eval_dataloaders:
                for camera_ray_bundle, batch in fixed_indices_eval_dataloader:
                    # time this the following line
                    inner_start = time()
                    height, width = camera_ray_bundle.shape
                    num_rays = height * width
                    outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, batch)
                    metrics_dict, _ = self.model.get_image_metrics_and_images(outputs, batch)
                    assert "num_rays_per_sec" not in metrics_dict
                    metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                    fps_str = "fps"
                    assert fps_str not in metrics_dict
                    metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
                    metrics_dict_list.append(metrics_dict)
                    progress.advance(task)
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            metrics_dict[key] = float(
                torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
            )
        self.train()
        return metrics_dict

    def load_pipeline(self, loaded_state: Dict[str, Any]) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
        """
        # TODO questionable if this going to work
        state = {key.replace("module.", ""): value for key, value in loaded_state.items()}
        self.load_state_dict(state, strict=False)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> typing.List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        datamanager_callbacks = self.datamanager.get_training_callbacks(training_callback_attributes)
        model_callbacks = self.model.get_training_callbacks(training_callback_attributes)
        callbacks = datamanager_callbacks + model_callbacks
        return callbacks

    def get_param_groups(self) -> Dict[str, typing.List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """
        datamanager_params = self.datamanager.get_param_groups()
        model_params = self.model.get_param_groups()
        # TODO(ethan): assert that key names don't overlap
        return {**datamanager_params, **model_params}
