from typing import List
from torch.utils.data import Dataset

from nerfstudio.data.datasets.base_dataset import InputDataset


class NesfDataset(Dataset):
    def __init__(self, datasets: List[InputDataset], main_set: int = 0):
        super().__init__()
        self._datasets: List[InputDataset] = datasets
        self.current_set: int = main_set
        
    @property 
    def has_masks(self):
        return self._datasets[self.current_set]._dataparser_outputs.has_masks
    
    @property
    def scale_factor(self):
        return self._datasets[self.current_set].scale_factor
    
    @property
    def scene_box(self):
        return self._datasets[self.current_set].scene_box
    
    @property
    def metadata(self):
        return self._datasets[self.current_set].metadata
    
    @property
    def cameras(self):
        return self._datasets[self.current_set].cameras
    
    def __len__(self):
        return len(self._datasets[self.current_set])
    
    def __iter__(self):
        return iter(self._datasets)
    
    def __getitem__(self, index) -> InputDataset:
        return self._datasets[self.current_set][index]
    
    def get_numpy_image(self, image_idx: int):
        return self._datasets[self.current_set].get_numpy_image(image_idx)
    
    def get_image(self, image_idx: int):
        return self._datasets[self.current_set].get_image(image_idx)
    
    def get_data(self, dataset_idx: int):
        return self._datasets[dataset_idx].get_data()
    
    def get_metadata(self, dataset_idx: int):
        return self._datasets[dataset_idx].get_metadata()
    
    def image_filenames(self, dataset_idx: int):
        return self._datasets[dataset_idx].image_filenames()
    
    def set_current_set(self, dataset_idx: int):
        assert dataset_idx < len(self._datasets)
        assert dataset_idx >= 0
        
        self.current_set = dataset_idx
        
    def get_set(self, dataset_idx: int) -> InputDataset:
        assert abs(dataset_idx) < len(self._datasets)
        return self._datasets[dataset_idx]
        
    def set_count(self) -> int:
        return len(self._datasets)