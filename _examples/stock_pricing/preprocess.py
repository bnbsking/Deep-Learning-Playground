from collections import Counter
from dataclasses import dataclass
import os
import random

import numpy as np
import pandas as pd


@dataclass
class Config:
    input_path = "/Users/james.chao/Desktop/codeMore/mygithub/Deep-Learning-Playground/_examples/stock_pricing/dataset/portfolio_data.csv"
    output_folder = "/Users/james.chao/Desktop/codeMore/mygithub/Deep-Learning-Playground/_examples/stock_pricing/dataset_preprocessed"
    window_size = 64
    seed = 0
    num_input_features = 4


# initialization
cfg = Config()
os.makedirs(cfg.output_folder, exist_ok=True)
random.seed(cfg.seed)


# time and log transformation
df = pd.read_csv(cfg.input_path)
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")
df = df.apply(np.log)


# sliding window
dataset = []
for i in range(len(df) - cfg.window_size):
    one_data = df.iloc[i: i + cfg.window_size].copy()
    # shape = (window_size, num_features) = (64, 4)
    for col in df.columns:
        one_data[f"y_{col}"] = df.iloc[i + cfg.window_size][col]
    one_data["series_id"] = i
    # shape = (window_size, num_features * 2 + 1) = (64, 9)
    dataset.append(one_data)
dataset = pd.concat(dataset, axis=0)
print(dataset.shape)


# save description
dataset_describe = dataset.describe()
dataset_describe.to_csv(os.path.join(cfg.output_folder, "dataset_describe.csv"))


# normalization
dataset_norm = (dataset - dataset_describe.loc["mean"]) / dataset_describe.loc["std"]
dataset_norm["series_id"] = dataset["series_id"]
print(dataset_norm.shape)


# splitting
n_all = len(dataset) // cfg.window_size
n_train = int(n_all * 0.8)
train_indices = random.sample(list(range(n_all)), n_train)
valid_indices = list(set(range(n_all)) - set(train_indices))
train_indices_df = pd.DataFrame({"series_id": train_indices})
valid_indices_df = pd.DataFrame({"series_id": valid_indices})
train_dataset = dataset.merge(train_indices_df, on="series_id")
valid_dataset = dataset.merge(valid_indices_df, on="series_id")
train_dataset_norm = dataset_norm.merge(train_indices_df, on="series_id")
valid_dataset_norm = dataset_norm.merge(valid_indices_df, on="series_id")
print(train_dataset_norm.shape, valid_dataset_norm.shape)


# numpy
train_x = train_dataset_norm.iloc[:, :cfg.num_input_features].to_numpy()
valid_x = valid_dataset_norm.iloc[:, :cfg.num_input_features].to_numpy()
train_x = train_x.reshape(-1, cfg.window_size, cfg.num_input_features)
valid_x = valid_x.reshape(-1, cfg.window_size, cfg.num_input_features)
#
train_y = train_dataset_norm.iloc[:, cfg.num_input_features: -1].to_numpy()
valid_y = valid_dataset_norm.iloc[:, cfg.num_input_features: -1].to_numpy()
train_y = train_y[::cfg.window_size, :]
valid_y = valid_y[::cfg.window_size, :]
#
print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)


# save
np.save(os.path.join(cfg.output_folder, "train_x.npy"), train_x)
np.save(os.path.join(cfg.output_folder, "train_y.npy"), train_y)
np.save(os.path.join(cfg.output_folder, "valid_x.npy"), valid_x)
np.save(os.path.join(cfg.output_folder, "valid_y.npy"), valid_y)
train_dataset.to_csv(os.path.join(cfg.output_folder, "train_dataset.csv"))
valid_dataset.to_csv(os.path.join(cfg.output_folder, "valid_dataset.csv"))
