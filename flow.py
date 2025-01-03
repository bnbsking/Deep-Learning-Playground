import json
import os
from typing import Dict, List

import torch
from tqdm import tqdm

from getter import get_metrics_pipeline
from type_defs import (
    DataLoaderCollections,
    Histories,
    HyperConfig,
    MetricsPipelineConfig,
    MonitorConfig,
    OutputCollectors,
)


def train(
        dataloaders: DataLoaderCollections,
        device: str,
        hyper: HyperConfig,
        model: torch.nn.Module,
        loss_func: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        output_collectors: OutputCollectors,
        histories: Histories,
        metrics_pipeline_cfg: MetricsPipelineConfig,
        monitor_cfgs: List[MonitorConfig],
    ):
    n_train = len(dataloaders.train.dataset)
    n_valid = len(dataloaders.valid.dataset)

    for epoch in tqdm(range(hyper.epochs)):
        # training loop
        train_loss = 0.
        for x, y in dataloaders.train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()      # 1. zero the parameter gradients
            pred = model(x)            # 2. f(x) and f'(x)
            loss = loss_func(pred, y)  # 3. compute loss
            loss.backward()            # 4. send loss to torch
            optimizer.step()           # 5. update model parameters by torch loss 
            scheduler.step()

            train_loss += loss.item() / n_train
            output_collectors.train.update(y, pred)
            
        # validation
        valid_loss = 0.
        for x, y in dataloaders.valid:
            with torch.no_grad():
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = loss_func(pred, y)
                valid_loss += loss.item() / n_valid
                output_collectors.valid.update(y, pred)
        
        # metrics
        output_collectors.train.postprocess()
        output_collectors.valid.postprocess()
        metrics_pipeline_train = get_metrics_pipeline(
            output_collectors.train,
            metrics_pipeline_cfg,
        )
        metrics_pipeline_valid = get_metrics_pipeline(
            output_collectors.valid,
            metrics_pipeline_cfg,
        )
        metrics_train = metrics_pipeline_train.run() | {"loss": train_loss}
        metrics_valid = metrics_pipeline_valid.run() | {"loss": valid_loss}
        histories.train.update(metrics_train)
        histories.valid.update(metrics_valid)
        output_collectors.train.reset()
        output_collectors.valid.reset()

        # save history
        with open(os.path.join(hyper.save_dir, "history.json"), "w") as f:
            history_dict = {
                "train": histories.train.history,
                "valid": histories.valid.history,
            }
            json.dump(history_dict, f, indent=4)
        
        # save ckpt
        for monitor_cfg in monitor_cfgs:
            hist_list = getattr(histories, monitor_cfg.mode).history[monitor_cfg.log_name]
            if monitor_cfg.target == "max" and hist_list[-1] == max(hist_list) \
                or monitor_cfg.target == "min" and hist_list[-1] == min(hist_list):
                torch.save(
                    model.state_dict(),
                    os.path.join(hyper.save_dir, f"best_{monitor_cfg.log_name}.pt")
                )
        if (epoch + 1) % hyper.save_ckpts_per_epochs == 0:
            torch.save(
                model.state_dict(),
                os.path.join(hyper.save_dir, f"epoch_{str(epoch).zfill(3)}.pt")
            )


def valid(
        dataloaders: DataLoaderCollections,
        device: str,
        hyper: HyperConfig,
        model: torch.nn.Module,
        loss_func: torch.nn.Module,
        output_collectors: OutputCollectors,
        histories: Histories,
        metrics_pipeline_cfg: MetricsPipelineConfig,
    ):
    n_valid = len(dataloaders.valid.dataset)

    # validation
    valid_loss = 0.
    for x, y in dataloaders.valid:
        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_func(pred, y)
            valid_loss += loss.item() / n_valid
            output_collectors.valid.update(y, pred)
    
    # metrics
    output_collectors.valid.postprocess()
    metrics_pipeline_valid = get_metrics_pipeline(
        output_collectors.valid,
        metrics_pipeline_cfg,
    )
    metrics_valid = metrics_pipeline_valid.run() | {"loss": valid_loss}
    histories.valid.update(metrics_valid)
    output_collectors.valid.save(hyper.save_dir, "valid")
    output_collectors.valid.reset()

    # save history
    with open(os.path.join(hyper.save_dir, "history_validation.json"), "w") as f:
        history_dict = {
            "valid": histories.valid.history,
        }
        json.dump(history_dict, f, indent=4)
    

def infer(
        dataloaders: DataLoaderCollections,
        device: str,
        hyper: HyperConfig,
        model: torch.nn.Module,
        output_collectors: OutputCollectors,
    ):
    # inference
    for x, y in dataloaders.infer:
        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            pred = model(x)
            output_collectors.infer.update(y, pred)
    
    # collect outputs
    output_collectors.infer.postprocess()
    output_collectors.infer.save(hyper.save_dir, "infer")
    output_collectors.infer.reset()
