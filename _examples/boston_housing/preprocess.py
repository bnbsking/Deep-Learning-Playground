from dataclasses import dataclass
import os
import random

import numpy as np
import pandas as pd


@dataclass
class Config:
    input_path = "/Users/james.chao/Desktop/codeMore/mygithub/Deep-Learning-Playground/examples/boston_housing/dataset/housing.csv"
    output_folder = "/Users/james.chao/Desktop/codeMore/mygithub/Deep-Learning-Playground/examples/boston_housing/dataset_preprocessed"
    std_multiple = 3
    seed = 0


# initialization
cfg = Config()
os.makedirs(cfg.output_folder, exist_ok=True)
random.seed(cfg.seed)


# format dataset
dataset = []
with open(cfg.input_path, "r") as f:
    for line in f.readlines():
        line_split_list = line.split(" ")
        line_split_list = [line_split for line_split in line_split_list if line_split]
        line_split_list = [float(line_split.replace("\n", "")) for line_split in line_split_list]
        assert len(line_split_list) == 14
        dataset.append(line_split_list)
dataset = pd.DataFrame(dataset)
print(dataset.shape)
print(dataset.head())


# filter outliers
dataset_mask = dataset.copy().astype(bool)
for col in dataset.columns:
    mean = dataset[col].mean()
    std = dataset[col].std()
    dataset_mask[col] = np.abs(dataset[col] - mean) / std < cfg.std_multiple
print("keep_rate_of_each_column:\n", dataset_mask.sum(axis=0) / len(dataset_mask))
row_mask = dataset_mask.all(axis=1)
print("keep_rate_of_all_data:\n", row_mask.sum() / len(row_mask))
dataset = dataset[row_mask]
print(dataset)


# correlation analysis and drop
corr = dataset.corr().abs()
is_const_col = corr.isna().sum(axis=1) == 14
dataset = dataset.loc[:, ~is_const_col]
print(dataset)


# normalization
dataset_describe = dataset.describe()
print(dataset_describe)
dataset_describe.to_csv(os.path.join(cfg.output_folder, "dataset_describe.csv"))


# keep raw data
dataset_norm = (dataset - dataset_describe.loc["mean"]) / dataset_describe.loc["std"]
print(dataset_norm)


# splitting
n_train = int(len(dataset) * 0.8)
train_indices = random.sample(list(range(len(dataset))), n_train)
valid_indices = list(set(range(len(dataset))) - set(train_indices))
train_dataset = dataset.iloc[train_indices]
valid_dataset = dataset.iloc[valid_indices]
train_dataset_norm = dataset_norm.iloc[train_indices]
valid_dataset_norm = dataset_norm.iloc[valid_indices]
print(train_dataset_norm.shape, valid_dataset_norm.shape)


# change to numpy
train_x = train_dataset_norm.iloc[:, :-1].to_numpy()
train_y = train_dataset_norm.iloc[:, -1:].to_numpy()
valid_x = valid_dataset_norm.iloc[:, :-1].to_numpy()
valid_y = valid_dataset_norm.iloc[:, -1:].to_numpy()
print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)


# save
np.save(os.path.join(cfg.output_folder, "train_x.npy"), train_x)
np.save(os.path.join(cfg.output_folder, "train_y.npy"), train_y)
np.save(os.path.join(cfg.output_folder, "valid_x.npy"), valid_x)
np.save(os.path.join(cfg.output_folder, "valid_y.npy"), valid_y)
train_dataset.to_csv(os.path.join(cfg.output_folder, "train_dataset.csv"))
valid_dataset.to_csv(os.path.join(cfg.output_folder, "valid_dataset.csv"))
