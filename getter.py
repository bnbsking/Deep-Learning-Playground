from typing import Dict, List, Optional

import torch

from data import datasets as ds, dataloaders as dl
from modules import (
    loss,
    metrics,
    models,
    optimizers,
    lr_schedulers,
)
from modules import (
    histories as hist,
    pretrain_loaders,
    output_collectors as oc,
    metrics
)
from type_defs import (
    DatasetConfig,
    DatasetCollections,
    DataLoaderConfig,
    DataLoaderCollections,
    Histories,
    MetricsPipelineConfig,
    OutputCollectors,
)


def get_datasets(dataset_cfgs: List[DatasetConfig]) -> DatasetCollections:
    datasets = DatasetCollections()
    for dataset_cfg in dataset_cfgs:
        dataset = getattr(ds, dataset_cfg.class_name)(**dataset_cfg.args)
        getattr(datasets, dataset_cfg.mode)[dataset_cfg.log_name] = dataset
    return datasets


def get_dataloaders(
        datasets: DatasetCollections,
        dataloader_cfgs: List[DataLoaderConfig],
    ) -> DataLoaderCollections:
    dataloaders = DataLoaderCollections()
    for dataloader_cfg in dataloader_cfgs:
        same_mode_dataset_dict = getattr(datasets, dataloader_cfg.mode)
        dataloader = getattr(dl, dataloader_cfg.class_name)(
            same_mode_dataset_dict,
            **dataloader_cfg.args
        )
        setattr(dataloaders, dataloader_cfg.mode, dataloader)
    return dataloaders


def get_model(model_cfg: Dict, device: str) -> torch.nn.Module:
    model = getattr(models, model_cfg["class_name"])(**model_cfg["args"])
    model = model.to(device)
    return model


def load_pretrain(
        model: torch.nn.Module,
        pretrain_loader_cfg: Optional[Dict] = None,
    ) -> torch.nn.Module:
    return getattr(pretrain_loaders, pretrain_loader_cfg["func_name"])(
            model,
            **pretrain_loader_cfg["args"]
        )


def get_loss(loss_cfg: Dict, device: str) -> torch.nn.Module:
    cls_name = loss_cfg["class_name"]
    if cls_name.startswith("torch.nn."):
        loss_cls = getattr(torch.nn, cls_name.replace("torch.nn.", ""))
    else:
        loss_cls = getattr(loss, cls_name)
    return loss_cls(**loss_cfg["args"]).to(device)


def get_optimizer(model: torch.nn.Module, optimizer_cfg: Dict) -> torch.optim.Optimizer:
    cls_name = optimizer_cfg["class_name"]
    if cls_name.startswith("torch.optim."):
        optimizer_cls = getattr(torch.optim, cls_name.replace("torch.optim.", ""))
    else:
        optimizer_cls = getattr(optimizers, cls_name)
    return optimizer_cls(model.parameters(), **optimizer_cfg["args"])


def get_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_cfg: Dict
    ) -> torch.optim.lr_scheduler:
    cls_name = scheduler_cfg["class_name"]
    if cls_name.startswith("torch.optim.lr_scheduler."):
        scheduler_cls = getattr(
            torch.optim.lr_scheduler,
            cls_name.replace("torch.optim.lr_scheduler.", "")
        )
    else:
        scheduler_cls = getattr(lr_schedulers, cls_name)
    return scheduler_cls(optimizer, **scheduler_cfg["args"])


def get_output_collectors(output_collector_cfgs: Dict) -> OutputCollectors:
    output_collectors = OutputCollectors()
    for output_collector_cfg in output_collector_cfgs:
        output_collector = getattr(oc, output_collector_cfg["class_name"])(
            **output_collector_cfg
        )
        setattr(output_collectors, output_collector_cfg["mode"], output_collector)
    return output_collectors


def get_histories(history_cfgs: Dict) -> Histories:
    histories = Histories()
    for history_cfg in history_cfgs:
        history = getattr(hist, history_cfg["class_name"])(**history_cfg["args"])
        setattr(histories, history_cfg["mode"], history)
    return histories


def get_metrics_pipeline(
        output_collector: oc.BaseOutputCollector,
        metrics_pipeline_cfg: MetricsPipelineConfig
    ) -> metrics.BaseMetricsPipeline:
    metrics_pipeline = getattr(metrics, metrics_pipeline_cfg.class_name)(
        output_collector,
        **metrics_pipeline_cfg.args
    )
    return metrics_pipeline
