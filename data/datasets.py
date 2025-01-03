from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import transforms


class LoadNpyDataset(Dataset):
    def __init__(
            self,
            npy_path_x: str,
            npy_path_y: Optional[str] = None,
            dtype_x: str = "float32",
            dtype_y: str = "float32",
            transform: Optional[str] = None,
            transform_args: Optional[dict] = None
        ):
        """
        Notes:
            - shape of normal data
                x: (num_data, num_input_features)
                y: (num_data, num_output_features) or dummy arr
            - shape of times series data
                x: (num_data, window_size, num_input_features) 
                y: (num_data, window_size, num_output_features) or dummy arr
            - shape of image data
                x: (num_data, height, width, num_channels)
                y: (num_data, height, width, num_channels) or dummy arr
        """
        self.x = np.load(npy_path_x, allow_pickle=True)
        if npy_path_y:
            self.y = np.load(npy_path_y, allow_pickle=True)
            assert len(self.x) == len(self.y)
        else:
            self.y = np.zeros((len(self.x), 1)) - 1
        if transform is not None:
            self.transform = getattr(transforms, transform)(**transform_args)
        else:
            self.transform = lambda x: x
        self.dtype_x = getattr(torch, dtype_x)
        self.dtype_y = getattr(torch, dtype_y)
    
    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.transform(self.x[index])
        x = torch.tensor(x, dtype=self.dtype_x)
        y = torch.tensor(self.y[index], dtype=self.dtype_y)
        return x, y
