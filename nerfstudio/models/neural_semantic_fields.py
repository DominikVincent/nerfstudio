from dataclasses import dataclass, field
from typing import Type, Dict, List, Tuple

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
        raise NotImplementedError

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        # TODO get Unet Parameters here
        raise NotImplementedError

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        return []

    def get_outputs(self, ray_bundle: RayBundle):
        # TODO implement UNET
        # TODO query NeRF
        # TODO do feature conversion + MLP
        # TODO do semantic rendering
        raise NotImplementedError

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

