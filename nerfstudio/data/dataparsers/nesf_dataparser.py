# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Data parser for nerfstudio datasets. """

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Type

from rich.console import Console

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.base_model import Model
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.utils.io import load_from_json

CONSOLE = Console(width=120)
MAX_AUTO_RESOLUTION = 1600


# TODO delete if not needed
# @dataclass
# class NesfDataparserOutputs:
#     """Dataparser outputs for the which will be used by the DataManager
#     for creating RayBundle and RayGT objects."""
#
#     image_filenames: List[List[Path]]
#     """Filenames for the images."""
#     cameras: List[Cameras]
#     """Camera object storing collection of camera information in dataset."""
#     alpha_color: Optional[List[TensorType[3]]] = None
#     """Color of dataset background."""
#     scene_box: List[SceneBox] = field(default_factory=lambda: [SceneBox()])
#     """Scene box of dataset. Used to bound the scene or provide the scene scale depending on model."""
#     mask_filenames: Optional[List[List[Path]]] = None
#     """Filenames for any masks that are required"""
#     metadata: Dict[str, Any] = to_immutable_dict({})
#     """Dictionary of any metadata that be required for the given experiment.
#     Will be processed by the InputDataset to create any additional tensors that may be required.
#     """
#     dataparser_transform: List[TensorType[3, 4]] = torch.eye(4)[:3, :]
#     """Transform applied by the dataparser."""
#     dataparser_scale: List[float] = 1.0
#     """Scale applied by the dataparser."""
#
#     def as_dict(self) -> dict:
#         """Returns the dataclass as a dictionary."""
#         return vars(self)
#
#     def save_dataparser_transform(self, path: Path):
#         """Save dataparser transform to json file. Some dataparsers will apply a transform to the poses,
#         this method allows the transform to be saved so that it can be used in other applications.
#
#         Args:
#             path: path to save transform to
#         """
#         data = {
#             "transform": [transform.tolist() for transform in self.dataparser_transform],
#             "scale": [float(scale) for scale in self.dataparser_scale],
#         }
#         if not path.parent.exists():
#             path.parent.mkdir(parents=True)
#         with open(path, "w", encoding="UTF-8") as file:
#             json.dump(data, file, indent=4)


@dataclass
class NesfDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: Nesf)

    data_config: Path = Path("")
    """Path to the config of the Nesf data. It's a json {config:[{model_config: config, data_parser_config: config, load_step: 1, load_dir:1}, 
    ...]}"""


def _load_model(load_dir: Path, load_step: int, data_dir: Path, config: Dict) -> Model:
    """
    Loads the model from a path via a Trainer and then extracts the model from the pipeline. The rest of the
    trainer gets ignored.

    TODO load just the model from the state dict instead of having trainer
    TODO use inference mode to setup trainer, but it has a bug
    TODO make it customizable by using the config of the model


    :param load_dir: dir of the model
    :param load_step: checkpoint step of the model
    :param data_dir: dir of where the data of the model is located
    :param config: model config. Not used yet
    :return: a loaded model.
    """
    train_config = TrainerConfig(
        method_name="nerfacto",
        experiment_name="tmp",
        data=data_dir,
        output_dir=Path("/tmp"),
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
                ),

            ),
            model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
        },
        vis="tensorboard",
        load_dir=load_dir,
        load_step=load_step
    )

    train_config.set_timestamp()
    train_config.pipeline.datamanager.dataparser.data = train_config.data
    train_config.save_config()

    trainer = train_config.setup(local_rank=0, world_size=1)
    trainer.setup()

    pipeline = trainer.pipeline
    model = pipeline.model

    return model


@dataclass
class Nesf(DataParser):
    """Nerfstudio DatasetParser"""

    config: NesfDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        # pylint: disable=too-many-statements

        if self.config.data_config.suffix == ".json":
            data_config = load_from_json(self.config.data_config)
        else:
            data_config = load_from_json(self.config.data_config / "data_config.json")

        models = []
        data_parser_outputs = []
        for conf in data_config["config"]:
            nerfstudio = NerfstudioDataParserConfig(**conf["data_parser_config"]).setup()
            dataparser_output = nerfstudio.get_dataparser_outputs()
            models.append({
                "load_dir": conf["load_dir"],
                "load_step": conf["load_step"],
                "data_parser": nerfstudio
            })
            # TODO maybe load model

            # parent path of file
            data_path = dataparser_output.image_filenames[0].parent.resolve()
            model = _load_model(load_dir=conf["load_dir"],
                                load_step=conf["load_step"],
                                data_dir=data_path,
                                config=conf["model_config"]
                                )

            # TODO update dataparser_output.metadata with model
            dataparser_output.metadata.update({"model": model})

            data_parser_outputs.append(dataparser_output)

        # TODO remove if uneeded
        # dataparser_outputs = NesfDataparserOutputs([data_parser_output.image_filenames for data_parser_output in data_parser_outputs],
        #     cameras=[data_parser_output.cameras for data_parser_output in data_parser_outputs],
        #     scene_box=[data_parser_output.scene_box for data_parser_output in data_parser_outputs],
        #     mask_filenames=[data_parser_output.mask_filenames if len(data_parser_output.mask_filenames) > 0 else None for data_parser_output in data_parser_outputs],
        #     dataparser_scale=[data_parser_output.dataparser_scale for data_parser_output in data_parser_outputs],
        #     dataparser_transform=[data_parser_output.dataparser_scale for data_parser_output in data_parser_outputs],
        #     metadata={
        #         # TODO safe model here
        #     },
        # )
        return data_parser_outputs
