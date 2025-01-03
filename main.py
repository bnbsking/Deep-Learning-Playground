import argparse
import os

import torch
import yaml

import flow
from getter import (
    get_datasets,
    get_dataloaders,
    get_histories,
    get_loss,
    get_model,
    get_optimizer,
    get_output_collectors,
    get_scheduler,
    load_pretrain,
)
from type_defs import (
    DatasetCollections,
    DatasetConfig,
    DataLoaderConfig,
    DataLoaderCollections,
    MetricsPipelineConfig,
    MonitorConfig,
    Histories,
    HyperConfig,
    OutputCollectors,
)


# configurations
parser = argparse.ArgumentParser()
parser.add_argument("--mode", "-o", type=str, choices=["train", "valid", "infer"], required=True)
parser.add_argument("--cfg", "-c", type=str, default="_examples/boston_housing/cfg.yaml")
args = parser.parse_args()
with open(args.cfg, 'r') as f:
    cfg = yaml.safe_load(f)


# initialization
dataset_cfgs = [DatasetConfig(**dataset_cfg) for dataset_cfg in cfg["datasets"]]
dataloader_cfgs = [DataLoaderConfig(**dataloader_cfg) for dataloader_cfg in cfg["dataloaders"]]
hyper = HyperConfig(**cfg["hyperparameters"])
if hyper.device in ("auto", "gpu", "cuda") and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
os.makedirs(hyper.save_dir, exist_ok=True)


# getter
datasets: DatasetCollections = get_datasets(dataset_cfgs)
dataloaders: DataLoaderCollections = get_dataloaders(datasets, dataloader_cfgs)
model: torch.nn.Module = get_model(cfg["model"], device)
model = load_pretrain(model, cfg["pretrain_loader"])
loss_func: torch.nn.Module = get_loss(cfg["loss"], device)
optimizer = get_optimizer(model, cfg["optimizer"])
scheduler = get_scheduler(optimizer, cfg["lr_scheduler"])
output_collectors: OutputCollectors = get_output_collectors(cfg["output_collectors"])
histories: Histories = get_histories(cfg["histories"])
metrics_pipeline_cfg = MetricsPipelineConfig(**cfg["metrics_pipeline"])
monitor_cfgs = [MonitorConfig(**mcfg) for mcfg in cfg["monitors"]]


# core
if args.mode == "train":
    getattr(flow, hyper.train_func_name)(
        dataloaders = dataloaders,
        device = device,
        hyper = hyper,
        model = model,
        loss_func = loss_func,
        optimizer = optimizer,
        scheduler = scheduler,
        output_collectors = output_collectors,
        histories = histories,
        metrics_pipeline_cfg = metrics_pipeline_cfg,
        monitor_cfgs = monitor_cfgs,
    )
elif args.mode == "valid":
    getattr(flow, hyper.valid_func_name)(
        dataloaders = dataloaders,
        device = device,
        hyper = hyper,
        model = model,
        loss_func = loss_func,
        output_collectors = output_collectors,
        histories = histories,
        metrics_pipeline_cfg = metrics_pipeline_cfg,
    )
elif args.mode == "infer":
    getattr(flow, hyper.infer_func_name)(
        dataloaders = dataloaders,
        device = device,
        hyper = hyper,
        model = model,
        output_collectors = output_collectors,
    )


print(f"{args.mode} finished.")
