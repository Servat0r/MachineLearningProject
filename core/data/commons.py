# Commons datasets and dataloaders: MONKs, ML-CUP, MNIST
from __future__ import annotations
import math
import os

import numpy as np
import pandas as pd
import gzip
from ..utils import *
from .datasets import *
from .dataloading import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


class MNIST(BaseDataset):

    IMAGE_SIZE = 28

    def __init__(
            self, base_dir: str, train: bool = True, flatten: bool = False,
    ):  # todo add also downlaod
        self.base_dir = base_dir
        self.train = train
        if train:
            img_folder = os.path.join(self.base_dir, 'train-images-idx3-ubyte.gz')
            labels_folder = os.path.join(self.base_dir, 'train-labels-idx1-ubyte.gz')
            self.images = gzip.open(img_folder, 'r')
            self.labels = gzip.open(labels_folder, 'r')
        else:
            img_folder = os.path.join(self.base_dir, 't10k-images-idx3-ubyte.gz')
            labels_folder = os.path.join(self.base_dir, 't10k-labels-idx1-ubyte.gz')
            self.images = gzip.open(img_folder, 'r')
            self.labels = gzip.open(labels_folder, 'r')
        self.images.read(16)
        self.labels.read(8)    # todo check!
        # Read image data into numpy arrays
        buf = self.images.read(self.IMAGE_SIZE * self.IMAGE_SIZE * len(self))
        self.image_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        if flatten:
            self.image_data = self.image_data.reshape((len(self), 1, self.IMAGE_SIZE ** 2))
        self.image_data = self.image_data.reshape((len(self), self.IMAGE_SIZE, self.IMAGE_SIZE))
        # Do the same for labels data
        buf = self.labels.read(len(self))
        self.labels_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)

    def __len__(self):
        return 60_000 if self.train else 10_000

    def __getitem__(self, item):
        return self.image_data[item], self.labels_data[item]

    def get_batch(self, indices: int | Iterable[int]):
        return np.take(self.image_data, indices, axis=0), np.take(self.labels_data, indices, axis=0)


def read_monk(name, dirpath: str = '../datasets/monks', shuffle=True, validation_size=None, dtype=np.float64):
    """
    Reads the monks datasets
    :param name: name of the dataset
    :param dirpath: Path of the directory of MONKS datasets.
    :param shuffle: If True, shuffles the dataset before returning. Defaults to True.
    :param validation_size: If not None, splits the dataset into a train and validation one
    with given validation size. Defaults to None.
    :param dtype: Numpy datatype to which datasets input will be transformed.
    Defaults to np.float64.
    :return: Train dataset as ArrayDataset as first value, and as second one a validation dataset
    if validation_size != None, otherwise None.
    """
    # read the dataset
    col_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
    fpath = os.path.join(dirpath, str(name))
    monk_dataset = pd.read_csv(fpath, sep=' ', names=col_names)
    monk_dataset.set_index('Id', inplace=True)
    labels = monk_dataset.pop('class')

    monk_dataset = OneHotEncoder().fit_transform(monk_dataset).toarray().astype(dtype)
    labels = labels.to_numpy()[:, np.newaxis]

    if validation_size is not None:
        train_monk_dataset, eval_monk_dataset, train_labels, eval_labels = train_test_split(
            monk_dataset, labels, test_size=validation_size, random_state=0, shuffle=shuffle, stratify=labels
        )
        train_monk_dataset = np.expand_dims(train_monk_dataset, axis=1)
        eval_monk_dataset = np.expand_dims(eval_monk_dataset, axis=1)
        train_labels = np.expand_dims(train_labels, axis=1)
        eval_labels = np.expand_dims(eval_labels, axis=1)
        # We return arrays instead of an ArrayDataset to allow k-fold and other techniques
        return train_monk_dataset, train_labels, eval_monk_dataset, eval_labels
        # return ArrayDataset(train_monk_dataset, train_labels), ArrayDataset(eval_monk_dataset, eval_labels)
    else:
        if shuffle:
            indexes = list(range(len(monk_dataset)))
            np.random.shuffle(indexes)
            monk_dataset = monk_dataset[indexes]
            labels = labels[indexes]

        monk_dataset = np.expand_dims(monk_dataset, axis=1)
        labels = np.expand_dims(labels, axis=1)
        return monk_dataset, labels, None, None
        # return ArrayDataset(monk_dataset, labels), None


def read_cup(
        int_ts=False, dirpath: str = '../datasets/cup', int_ts_size=0.2, shuffle=True,
        validation_size=3/8, dtype=np.float64,
):
    """
    Reads the CUP training and test set
    :return: CUP training data, CUP training targets and CUP test data (as numpy ndarray)
    """
    # read the dataset
    col_names = ['id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'target_x', 'target_y']
    int_ts_path = os.path.join(dirpath, "CUP-INTERNAL-TEST.csv")
    dev_set_path = os.path.join(dirpath, "CUP-DEV-SET.csv")
    orig_tr_path = os.path.join(dirpath, "ML-CUP22-TR.csv")
    test_set_path = os.path.join(dirpath, "ML-CUP22-TS.csv")

    # If internal test set is required and paths for it and development one do not exist, create them
    if int_ts and not (os.path.exists(int_ts_path) and os.path.exists(orig_tr_path)):
        df = pd.read_csv(orig_tr_path, sep=',', names=col_names, skiprows=range(7), usecols=range(0, 13))
        # Shuffle the dataframe by using ids
        df = df.sample(frac=1, axis=0, random_state=0)
        # Take the first (100*int_ts_size)% of the dataset to form internal test set
        int_ts_df = df.iloc[:math.floor(len(df) * int_ts_size), :]
        # Use the remaining part as development set
        dev_set_df = df.iloc[math.floor(len(df) * int_ts_size):, :]
        # Save extracted dataframes with 6-digit precision (as they are in original file)
        int_ts_df.to_csv(path_or_buf=int_ts_path, index=False, float_format='%.6f', header=False)
        dev_set_df.to_csv(path_or_buf=dev_set_path, index=False, float_format='%.6f', header=False)

    # If internal test set is required and paths for it and development one already exist, use them
    int_ts_data, int_ts_targets = None, None
    if int_ts and os.path.exists(int_ts_path) and os.path.exists(dev_set_path):
        tr_data = pd.read_csv(dev_set_path, sep=',', names=col_names, skiprows=range(7), usecols=range(1, 11))
        tr_targets = pd.read_csv(dev_set_path, sep=',', names=col_names, skiprows=range(7), usecols=range(11, 13))

        int_ts_data = pd.read_csv(int_ts_path,  sep=',', names=col_names, skiprows=range(7), usecols=range(1, 11))
        int_ts_targets = pd.read_csv(int_ts_path,  sep=',', names=col_names, skiprows=range(7), usecols=range(11, 13))

        int_ts_data = int_ts_data.to_numpy(dtype=dtype)
        int_ts_targets = int_ts_targets.to_numpy(dtype=dtype)
    # Either internal test set is not required or original training set has not been split, load original training set
    # ("fallback")
    else:
        tr_data = pd.read_csv(orig_tr_path, sep=',', names=col_names, skiprows=range(7), usecols=range(1, 11))
        tr_targets = pd.read_csv(orig_tr_path, sep=',', names=col_names, skiprows=range(7), usecols=range(11, 13))

    cup_ts_data = pd.read_csv(test_set_path, sep=',', names=col_names[: -2], skiprows=range(7), usecols=range(1, 11))

    tr_data = tr_data.to_numpy(dtype=dtype)
    tr_targets = tr_targets.to_numpy(dtype=dtype)
    cup_ts_data = cup_ts_data.to_numpy(dtype=dtype)

    if shuffle:
        indexes = np.arange(len(tr_targets))
        np.random.shuffle(indexes)
        # todo maybe this is better with np.take(out=tr_data/tr_targets)
        tr_data = tr_data[indexes]
        tr_targets = tr_targets[indexes]

    # if internal test set is not in a csv file and is required, extract it from already loaded original training set
    if int_ts and os.path.exists(dev_set_path) and not os.path.exists(int_ts_path):
        tr_data, int_ts_data, tr_targets, int_ts_targets = train_test_split(
            tr_data, tr_targets, test_size=int_ts_size, random_state=0,
        )

    # Split into train and validation todo Sure? Did it mean this or to use internal test set for validation?
    tr_data, ev_data, tr_targets, ev_targets = train_test_split(
        tr_data, tr_targets, test_size=validation_size, random_state=0
    )

    train_dataset = ArrayDataset(tr_data, tr_targets)
    eval_dataset = ArrayDataset(ev_data, ev_targets)
    int_ts_dataset = None if int_ts_data is None else ArrayDataset(int_ts_data, int_ts_targets)

    return train_dataset, eval_dataset, int_ts_dataset, cup_ts_data


__all__ = [
    'read_monk',
    'read_cup',
    'MNIST',
]
