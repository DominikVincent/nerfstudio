
"""
Nesf dataset.
"""

from typing import Dict

import torch

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs, Semantics
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_semantics_and_mask_tensors_from_path
from nerfstudio.models.base_model import Model


class NesfDataset(InputDataset):
    """Dataset that returns images and models TODO and semantics and masks.

    Args:
        dataparser_outputs: description of where and how to read input images.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        assert "model" in dataparser_outputs.metadata.keys() and isinstance(self.metadata["semantics"], Model)
        self.model = self.metadata["model"]

    def get_metadata(self, data: Dict) -> Dict:
        # TODO add semantics to metadata
        return {"model": self.model}
