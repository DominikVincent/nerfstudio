



"""
Dataset.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchtyping import TensorType

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path


class NesfDataset(Dataset):
    """Dataset that returns images.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs
    """

    def __init__(self, dataparser_outputs: List[DataparserOutputs], scale_factor: float = 1.0):
        super().__init__()
        self._dataparser_outputs = dataparser_outputs
        self.has_masks = dataparser_outputs[0].mask_filenames is not None
        self.scale_factor = scale_factor
        self.scene_box = [deepcopy(dataparser_output.scene_box) for dataparser_output in dataparser_outputs]
        self.metadata = [deepcopy(dataparser_output.metadata) for dataparser_output in dataparser_outputs]
        self.cameras = [deepcopy(dataparser_output.cameras) for dataparser_output in dataparser_outputs]
        [camera.rescale_output_resolution(scaling_factor=scale_factor) for camera in self.cameras]
        self.idx_model_table = {}
        self.semantics = "semantics" in self.metadata[0]

    def __len__(self):
        return sum([len(dataparser_output.image_filenames) for dataparser_output in self._dataparser_outputs])

    def _get_model_image_indices(self, image_idx: int) -> (int, int):
        if image_idx in self.idx_model_table:
            return self.idx_model_table[image_idx]
        model_idx = 0
        acc_sum = len(self._dataparser_outputs[0].image_filenames)
        while image_idx >= acc_sum:
            model_idx += 1
            acc_sum += len(self._dataparser_outputs[model_idx].image_filenames)

        image_idx = image_idx - acc_sum + len(self._dataparser_outputs[model_idx])

        self.idx_model_table[image_idx] = (model_idx, image_idx)
        return model_idx, image_idx

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        model_idx, image_idx = self._get_model_image_indices(image_idx)
        image_filename = self._dataparser_outputs[model_idx].image_filenames[image_idx]
        pil_image = Image.open(image_filename)
        if self.scale_factor != 1.0:
            width, height = pil_image.size
            newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
            pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image

    def get_image(self, image_idx: int) -> TensorType["image_height", "image_width", "num_channels"]:
        """Returns a 3 channel image.

        Args:
            image_idx: The image index in the dataset.
        """
        model_idx, _ = self._get_model_image_indices(image_idx)

        image = torch.from_numpy(self.get_numpy_image(image_idx).astype("float32") / 255.0)
        if self._dataparser_outputs[model_idx].alpha_color is not None and image.shape[-1] == 4:
            assert image.shape[-1] == 4
            image = image[:, :, :3] * image[:, :, -1:] + self._dataparser_outputs[model_idx].alpha_color * (1.0 - image[:, :, -1:])
        else:
            image = image[:, :, :3]
        return image

    def get_data(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        model_idx, image_sub_idx = self._get_model_image_indices(image_idx)

        image = self.get_image(image_idx)
        data = {"image_idx": image_idx}
        data["image"] = image
        if self.has_masks:
            mask_filepath = self._dataparser_outputs[model_idx].mask_filenames[image_sub_idx]
            data["mask"] = get_image_mask_tensor_from_path(filepath=mask_filepath, scale_factor=self.scale_factor)
            assert (
                data["mask"].shape[:2] == data["image"].shape[:2]
            ), f"Mask and image have different shapes. Got {data['mask'].shape[:2]} and {data['image'].shape[:2]}"
        metadata = self.get_metadata(data)
        data.update(metadata)
        return data

    # pylint: disable=no-self-use
    def get_metadata(self, data: Dict) -> Dict:
        """Method that can be used to process any additional metadata that may be part of the model inputs.

        Args:
            image_idx: The image index in the dataset.
        """
        if self.semantics:
            # TODO load semantic images
            pass

        # Load the model

        return {}

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.get_data(image_idx)
        return data

    @property
    def image_filenames(self) -> List[Path]:
        """
        Returns image filenames for this dataset.
        The order of filenames is the same as in the Cameras object for easy mapping.
        """

        return self._dataparser_outputs.image_filenames
