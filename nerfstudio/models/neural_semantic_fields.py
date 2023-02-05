from collections import defaultdict
from dataclasses import dataclass, field
from typing import Type, Dict, List, Tuple, Any

import torch
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import TrainingCallbackAttributes, TrainingCallback
from nerfstudio.models.base_model import ModelConfig, Model


@dataclass
class NeuralSemanticFieldConfig(ModelConfig):
    """Config for Neural Semantic field"""
    _target: Type = field(default_factory=lambda: NeuralSemanticFieldModel)


class NeuralSemanticFieldModel(Model):

    config: NeuralSemanticFieldConfig

    def populate_modules(self):
        # TODO create 3D-Unet here
        # raise NotImplementedError
        pass

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        # TODO get Unet Parameters here
        # raise NotImplementedError
        pass

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        return []

    def get_outputs(self, ray_bundle: RayBundle, batch: Dict[str, Any]):
        # TODO implement UNET
        # TODO query NeRF
        # TODO do feature conversion + MLP
        # TODO do semantic rendering

        # Query the NeRF model at the ray bundles
        value = torch.zeros((ray_bundle.shape.item(), 3))

        raise value

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        raise NotImplementedError

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
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

