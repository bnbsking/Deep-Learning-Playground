from typing import Dict
from torch.utils.data import Dataset, DataLoader


class DefaultLoader(DataLoader):
    def __init__(self, dataset_dict: Dict[str, Dataset], **kwargs):
        assert len(dataset_dict) == 1
        self.dataset = dataset_dict[next(iter(dataset_dict))]
        super().__init__(self.dataset, **kwargs)
