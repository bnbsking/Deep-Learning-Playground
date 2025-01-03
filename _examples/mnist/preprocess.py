from collections import Counter
from dataclasses import dataclass
import os
import random

import numpy as np
import pandas as pd


@dataclass
class Config:
    input_path_train = "/Users/james.chao/Desktop/codeMore/mygithub/Deep-Learning-Playground/_examples/mnist/dataset/mnist_train.csv"
    input_path_valid = "/Users/james.chao/Desktop/codeMore/mygithub/Deep-Learning-Playground/_examples/mnist/dataset/mnist_test.csv"
    output_folder = "/Users/james.chao/Desktop/codeMore/mygithub/Deep-Learning-Playground/_examples/mnist/dataset_preprocessed"
    sample_each_set = 1000
    seed = 0
    num_input_features = 4


# initialization
cfg = Config()
os.makedirs(cfg.output_folder, exist_ok=True)
random.seed(cfg.seed)


# load
df_train = pd.read_csv(cfg.input_path_train)
df_valid = pd.read_csv(cfg.input_path_valid)
print(df_train.shape, df_valid.shape)
print(df_train.head())


# subset sampling
df_train_sample = df_train.sample(cfg.sample_each_set, random_state=cfg.seed)
df_valid_sample = df_valid.sample(cfg.sample_each_set, random_state=cfg.seed)
print(
    df_train_sample.shape,
    df_valid.shape,
    df_train_sample["label"].value_counts(),
    df_valid_sample["label"].value_counts()
)


# numpy
train_x = df_train_sample.drop(columns=["label"]).values.reshape(-1, 28, 28)
train_y = df_train_sample["label"].values.reshape(-1, 1)
valid_x = df_valid_sample.drop(columns=["label"]).values.reshape(-1, 28, 28)
valid_y = df_valid_sample["label"].values.reshape(-1, 1)
print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)


# save
np.save(os.path.join(cfg.output_folder, "train_x.npy"), train_x)
np.save(os.path.join(cfg.output_folder, "train_y.npy"), train_y)
np.save(os.path.join(cfg.output_folder, "valid_x.npy"), valid_x)
np.save(os.path.join(cfg.output_folder, "valid_y.npy"), valid_y)
