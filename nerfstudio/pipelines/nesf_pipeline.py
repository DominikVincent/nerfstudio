from __future__ import annotations

import typing
from dataclasses import dataclass, field
from inspect import Parameter
from pathlib import Path
from time import time
from typing import Any, Dict, Optional, Type

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from rich.console import Console
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
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.datamanagers.nesf_datamanager import (
    NesfDataManager,
    NesfDataManagerConfig,
)
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.models.nesf import NeuralSemanticFieldConfig, NeuralSemanticFieldModel
from nerfstudio.pipelines.base_pipeline import (
    Pipeline,
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nerfstudio.utils import profiler, writer

CONSOLE = Console(width=120)


@dataclass
class NesfPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: NesfPipeline)
    """target class to instantiate"""
    datamanager: NesfDataManagerConfig = NesfDataManagerConfig()
    """specifies the datamanager config"""
    model: NeuralSemanticFieldConfig = NeuralSemanticFieldConfig()
    """specifies the model config"""
    images_per_all_evaluation = 15
    """how many images should be evaluated per scene when evaluating all images. -1 means all"""
    save_images = False
    """save images during all image evaluation"""
    images_to_sample_during_eval_image: int = 4


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
        print("### NesfPipeline: datamanager setup done.")
        self.datamanager.to(device)

        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_datasets is not None, "Missing input dataset"

        self._model: NeuralSemanticFieldModel = config.model.setup(
            scene_box=self.datamanager.train_datasets.get_set(0).scene_box,
            num_train_data=-1,
            metadata=self.datamanager.train_dataset.metadata,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                NeuralSemanticFieldModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
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
        self.datamanager.models_to_cpu(step)
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
        self.datamanager.models_to_cpu(step)
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
        if self.config.images_to_sample_during_eval_image > 1:
            image_idx, model_idx, camera_ray_bundle, batch = self.datamanager.next_eval_images(
                step, self.config.images_to_sample_during_eval_image
            )
            image_idx = image_idx[0]
            model_idx = model_idx[0]
            batch = batch[0]
        else:
            image_idx, model_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(step)

        batch["image_idx"] = image_idx
        batch["model_idx"] = model_idx
        outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, batch)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = image_idx
        metrics_dict["model_idx"] = model_idx
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = len(camera_ray_bundle)
        self.train()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_eval_image_metrics(self, step: Optional[int] = None, save_path: Optional[Path] = None, wandb=False):
        """Iterate over all the images in the eval dataset and get the average.

        Returns:
            metrics_dict: dictionary of metrics
            save_path: path to save the images to. if None, do not save images.
        """
        self.eval()
        metrics_dict_list = []
        num_images = (
            min(
                sum(
                    [
                        len(fixed_indices_eval_dataloader)
                        for fixed_indices_eval_dataloader in self.datamanager.fixed_indices_eval_dataloaders
                    ]
                ),
                len(self.datamanager.fixed_indices_eval_dataloaders) * self.config.images_per_all_evaluation,
            )
            if self.config.images_per_all_evaluation >= 0
            else 999999999
        )
        step = 0
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for model_idx, fixed_indices_eval_dataloader in enumerate(self.datamanager.fixed_indices_eval_dataloaders):
                ray_bundles = []
                for image_idx, (camera_ray_bundle, batch) in enumerate(fixed_indices_eval_dataloader):
                    if image_idx >= self.config.images_to_sample_during_eval_image - 1:
                        break
                    ray_bundles.insert(0, camera_ray_bundle)
                for i, (camera_ray_bundle, batch) in enumerate(fixed_indices_eval_dataloader):
                    ray_bundles.insert(0, camera_ray_bundle)
                    batch["model_idx"] = model_idx
                    batch["image_idx"] = batch["image_idx"]
                    print("model_idx", model_idx, "image_idx", batch["image_idx"])

                    if i >= self.config.images_per_all_evaluation and self.config.images_per_all_evaluation >= 0:
                        break
                    # time this the following line
                    inner_start = time()
                    height, width = camera_ray_bundle.shape
                    num_rays = height * width
                    outputs = self.model.get_outputs_for_camera_ray_bundle(ray_bundles, batch)
                    metrics_dict, image_dict = self.model.get_image_metrics_and_images(outputs, batch)
                    assert "num_rays_per_sec" not in metrics_dict
                    metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                    fps_str = "fps"
                    assert fps_str not in metrics_dict
                    metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
                    metrics_dict["image_idx"] = batch["image_idx"]
                    metrics_dict["model_idx"] = batch["model_idx"]
                    metrics_dict_list.append(metrics_dict)

                    img = image_dict["img"]
                    if wandb:
                        writer.put_image("test_image", img, step=step)
                        writer.put_dict("test_image", metrics_dict, step=step)
                        writer.write_out_storage()

                    img = img.cpu().numpy()
                    if save_path is not None:
                        file_path = save_path / f"{model_idx:03d}" / f"{batch['image_idx']:04d}.png"
                        # create the directory if it does not exist
                        if not file_path.parent.exists():
                            file_path.parent.mkdir(parents=True)
                        # save the image
                        img_pil = Image.fromarray((img * 255).astype(np.uint8))
                        img_pil.save(file_path)
                    ray_bundles.pop()
                    step += 1
                    progress.advance(task)
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if key == "image_idx" or key == "model_idx":
                continue
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
