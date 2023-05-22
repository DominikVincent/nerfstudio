from typing import Dict, List

import torch
from torch.utils.data import Dataset

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_semantics_and_mask_tensors_from_path
from nerfstudio.utils.nesf_utils import get_memory_usage


class NesfItemDataset(InputDataset):
    """This is an Input dataset which has the additional metadata information field.
    The meta data field contains the model of this dataset and potentially the semantic masks.
    Args:
        dataparser_outputs: description of where and how to read input data.
        scalefactor: The scaling factor for the dataparser outputs
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1):
        super().__init__(dataparser_outputs, scale_factor)
        assert "model" in dataparser_outputs.metadata
        self.model = dataparser_outputs.metadata["model"]
        self.semantics = dataparser_outputs.metadata["semantics"]
        self.mask_indices = torch.tensor(
            [self.semantics.classes.index(mask_class) for mask_class in self.semantics.mask_classes]
        ).view(1, 1, -1)
        print("NesfItemDataset - memory usage: ", get_memory_usage())

    def get_metadata(self, data: Dict) -> Dict:
        filepath = self.semantics.filenames[data["image_idx"]]
        semantic_label, mask = get_semantics_and_mask_tensors_from_path(
            filepath=filepath, mask_indices=self.mask_indices, scale_factor=self.scale_factor
        )
        if "mask" in data.keys():
            mask = mask & data["mask"]
        return {"model": self.model, "semantics": semantic_label}


class NesfDataset(Dataset):
    def __init__(self, datasets: List[NesfItemDataset], main_set: int = 0):
        super().__init__()
        self._datasets: List[NesfItemDataset] = datasets
        self.current_set_idx: int = main_set

    @property
    def has_masks(self):
        return self._datasets[self.current_set_idx].has_masks

    @property
    def scale_factor(self):
        return self._datasets[self.current_set_idx].scale_factor

    @property
    def scene_box(self):
        return self._datasets[self.current_set_idx].scene_box

    @property
    def metadata(self):
        return self._datasets[self.current_set_idx].metadata

    @property
    def cameras(self):
        return self._datasets[self.current_set_idx].cameras

    def __len__(self):
        return len(self._datasets[self.current_set_idx])

    def __iter__(self):
        return iter(self._datasets)

    def __getitem__(self, index) -> Dict:
        return self._datasets[self.current_set_idx][index]

    def get_numpy_image(self, image_idx: int):
        return self._datasets[self.current_set_idx].get_numpy_image(image_idx)

    def get_image(self, image_idx: int):
        return self._datasets[self.current_set_idx].get_image(image_idx)

    def get_data(self, image_idx: int):
        return self._datasets[self.current_set_idx].get_data(image_idx)

    def get_metadata(self, data: Dict):
        return self._datasets[self.current_set_idx].get_metadata(data)

    def image_filenames(self):
        return self._datasets[self.current_set_idx].image_filenames

    def set_current_set(self, dataset_idx: int):
        assert dataset_idx < len(self._datasets)
        assert dataset_idx >= 0

        self.current_set_idx = dataset_idx

    def get_set(self, dataset_idx: int) -> InputDataset:
        assert abs(dataset_idx) < len(self._datasets)
        return self._datasets[dataset_idx]

    def set_count(self) -> int:
        return len(self._datasets)

    @property
    def current_set(self) -> NesfItemDataset:
        return self._datasets[self.current_set_idx]
