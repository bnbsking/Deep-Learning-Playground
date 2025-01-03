from typing import Dict, List

import numpy as np
import torch
import torchvision.transforms as transforms


class SimpleTransform:
    def __init__(self, cfgs: List[Dict]):
        transform_list = [getattr(transforms, cfg['name'])(**cfg['args']) for cfg in cfgs]
        self.transform = transforms.Compose(transform_list)
    
    def __call__(self, x: np.ndarray) -> torch.Tensor:
        """
        Args:
            x (np.ndarray): shape=(H, W)
        Returns:
            (torch.Tensor): shape=(1, H, W)
        """
        return self.transform(x)
