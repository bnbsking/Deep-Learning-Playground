from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from torch.utils.data import Dataset, DataLoader
from modules.output_collectors import BaseOutputCollector
from modules.histories import BaseHistory


@dataclass
class DatasetConfig:
    mode: Literal["train", "valid", "test"]
    log_name: str
    class_name: str
    args: dict


@dataclass(frozen=False)
class DatasetCollections:
    train: Dict[str, Dataset] = field(default_factory=dict)
    valid: Dict[str, Dataset] = field(default_factory=dict)
    infer: Dict[str, Dataset] = field(default_factory=dict)


@dataclass
class DataLoaderConfig:
    mode: Literal["train", "valid", "infer"]
    class_name: str
    args: dict


@dataclass(frozen=False)
class DataLoaderCollections:
    train: Optional[DataLoader] = field(default_factory=lambda : None)
    valid: Optional[DataLoader] = field(default_factory=lambda : None)
    infer: Optional[DataLoader] = field(default_factory=lambda : None)


@dataclass
class HyperConfig:
    device: Literal["auto", "gpu", "cuda", "cpu"]
    epochs: int
    save_dir: str
    save_ckpts_per_epochs: int
    train_func_name: str = "train"
    valid_func_name: str = "valid"
    infer_func_name: str = "infer"


@dataclass(frozen=False)
class OutputCollectors:
    train: Optional[BaseOutputCollector] = field(default_factory=lambda : None)
    valid: Optional[BaseOutputCollector] = field(default_factory=lambda : None)
    infer: Optional[BaseOutputCollector] = field(default_factory=lambda : None)


@dataclass(frozen=False)
class Histories:
    train: Optional[BaseHistory] = field(default_factory=lambda : None)
    valid: Optional[BaseHistory] = field(default_factory=lambda : None)
    infer: Optional[BaseHistory] = field(default_factory=lambda : None)


@dataclass
class MetricsPipelineConfig:
    class_name: str
    args: Dict


@dataclass
class MonitorConfig:
    mode: Literal["train", "valid", "infer"]
    log_name: str
    target: Literal["max", "min"]
