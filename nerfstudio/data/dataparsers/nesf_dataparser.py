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

import math
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Optional, Type, List

import numpy as np
import torch
from PIL import Image
from rich.console import Console
from typing_extensions import Literal

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json

from __future__ import annotations

import json
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import torch
from torchtyping import TensorType

import nerfstudio.configs.base_config as cfg
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.scene_box import SceneBox


CONSOLE = Console(width=120)
MAX_AUTO_RESOLUTION = 1600



@dataclass
class NesfDataparserOutputs:
    """Dataparser outputs for the which will be used by the DataManager
    for creating RayBundle and RayGT objects."""

    image_filenames: List[List[Path]]
    """Filenames for the images."""
    cameras: List[Cameras]
    """Camera object storing collection of camera information in dataset."""
    alpha_color: Optional[List[TensorType[3]]] = None
    """Color of dataset background."""
    scene_box: List[SceneBox] = field(default_factory=lambda: [SceneBox()])
    """Scene box of dataset. Used to bound the scene or provide the scene scale depending on model."""
    mask_filenames: Optional[List[List[Path]]] = None
    """Filenames for any masks that are required"""
    metadata: Dict[str, Any] = to_immutable_dict({})
    """Dictionary of any metadata that be required for the given experiment.
    Will be processed by the InputDataset to create any additional tensors that may be required.
    """
    dataparser_transform: List[TensorType[3, 4]] = torch.eye(4)[:3, :]
    """Transform applied by the dataparser."""
    dataparser_scale: List[float] = 1.0
    """Scale applied by the dataparser."""

    def as_dict(self) -> dict:
        """Returns the dataclass as a dictionary."""
        return vars(self)

    def save_dataparser_transform(self, path: Path):
        """Save dataparser transform to json file. Some dataparsers will apply a transform to the poses,
        this method allows the transform to be saved so that it can be used in other applications.

        Args:
            path: path to save transform to
        """
        data = {
            "transform": [transform.tolist() for transform in self.dataparser_transform],
            "scale": [float(scale) for scale in self.dataparser_scale],
        }
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        with open(path, "w", encoding="UTF-8") as file:
            json.dump(data, file, indent=4)



@dataclass
class NesfDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: Nesf)

    data_config: Path = Path("")
    """Path to the config of the Nesf data. It's a json {config:[{model_config: config, load_step: 1, load_dir:1}, ...]}"""


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
            nerfstudio = NerfstudioDataParserConfig(**conf["model_config"]).setup()
            dataparser_output = nerfstudio.get_dataparser_outputs()
            models.append({
                "load_dir": conf["load_dir"],
                "load_step": conf["load_step"],
                "data_parser": nerfstudio
            })
            data_parser_outputs.append(dataparser_output)
            # TODO maybe load model

        dataparser_outputs = NesfDataparserOutputs(
            image_filenames=[data_parser_output.image_filenames for data_parser_output in data_parser_outputs],
            cameras=[data_parser_output.cameras for data_parser_output in data_parser_outputs],
            scene_box=[data_parser_output.scene_box for data_parser_output in data_parser_outputs],
            mask_filenames=[data_parser_output.mask_filenames if len(data_parser_output.mask_filenames) > 0 else None for data_parser_output in data_parser_outputs],
            dataparser_scale=[data_parser_output.dataparser_scale for data_parser_output in data_parser_outputs],
            dataparser_transform=[data_parser_output.dataparser_scale for data_parser_output in data_parser_outputs],
            metadata={
                # TODO safe model here
            },
        )
        return dataparser_outputs