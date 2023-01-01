# Commons datasets and dataloaders: MONKs, ML-CUP
from __future__ import annotations
import math
import os
import pandas as pd

from ..utils import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def read_monk(name, directory_path: str = '../datasets/monks', shuffle_once=True,
              shuffle_seed=0, validation_size=None, dtype=np.float32):
    """
    Reads the monks datasets
    :param name: name of the dataset
    :param directory_path: Path of the directory of MONKS datasets.
    :param shuffle_once: If True, shuffles the dataset before returning. Defaults to True.
    :param shuffle_seed: Seed for shuffling the dataset. Defaults to 0.
    :param validation_size: If not None, splits the dataset into a train and validation one
    with given validation size. Defaults to None.
    :param dtype: Numpy datatype to which datasets input will be transformed.
    Defaults to np.float32.
    :return: Train dataset as ArrayDataset as first value, and as second one a validation dataset
    if validation_size != None, otherwise None.
    """
    # read the dataset
    column_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
    file_path = os.path.join(directory_path, str(name))
    monk_dataset = pd.read_csv(file_path, sep=' ', names=column_names)
    monk_dataset.set_index('Id', inplace=True)
    labels = monk_dataset.pop('class')

    monk_dataset = OneHotEncoder().fit_transform(monk_dataset).toarray().astype(dtype)
    labels = labels.to_numpy()[:, np.newaxis]

    if validation_size is not None:
        train_monk_dataset, eval_monk_dataset, train_labels, eval_labels = train_test_split(
            monk_dataset, labels, test_size=validation_size, random_state=0, shuffle=shuffle_once, stratify=labels
        )
        # We return arrays instead of an ArrayDataset to allow k-fold and other techniques
        return train_monk_dataset, train_labels, eval_monk_dataset, eval_labels
    else:
        if shuffle_once:
            indexes = list(range(len(monk_dataset)))
            rng = np.random.default_rng(seed=shuffle_seed)
            rng.shuffle(indexes)
            monk_dataset = monk_dataset[indexes]
            labels = labels[indexes]
        return monk_dataset, labels, None, None


def read_cup(
        use_internal_test_set=False, directory_path: str = '../datasets/cup',
        internal_test_set_size=0.1, shuffle_once=True, shuffle_seed=0, dtype=np.float32,
):
    """
    Reads the CUP training and test set.
    :param use_internal_test_set: If True, loads and returns also Internal Test Set.
    If True and:
     - internal test set and development set CSV files are not existing, splits thw whole
     dataset into development and internal test set according to :param internal_test_set_size
     and saves them as CSV files;
     - internal test and development sets CSV files exist, loads them;
     - development set file exists and internal test set file is missing, detaches a portion
     of development set to be used as internal test set.
    Defaults to False.
    :param directory_path: Relative path of the directory containing CUP dataset.
    :param internal_test_set_size: Relative size of the internal test set w.r.t. the whole dataset.
    It has effect only if :param sue_internal_test_set = True and internal test set needs to be
    created or detached. Defaults to 0.1.
    :param shuffle_once: If True, shuffles training set before returning. Defaults to True.
    :param shuffle_seed: Seed for shuffling the dataset. Defaults to 0.
    :param dtype: Numpy datatype in which data shall be returned. Defaults to numpy.float32.
    :return: CUP training data, CUP training targets and CUP test data (as numpy ndarray)
    """
    # read the dataset
    column_names = ['id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'target_x', 'target_y']
    original_train_set_path = os.path.join(directory_path, "ML-CUP22-TR.csv")
    test_set_path = os.path.join(directory_path, "ML-CUP22-TS.csv")

    dataframe = pd.read_csv(
        original_train_set_path, sep=',', names=column_names, skiprows=range(7), usecols=range(1, 12)
    )
    # Convert dataframe to numpy
    data_array = dataframe.to_numpy(dtype=dtype)
    x, y = data_array[:, :9], data_array[:, 9:]
    cup_test_set_data = pd.read_csv(
        test_set_path, sep=',', names=column_names[: -2], skiprows=range(7), usecols=range(1, 10)
    ).to_numpy(dtype=dtype)
    if use_internal_test_set:
        x_dev, x_test, y_dev, y_test = train_test_split(
            x, y, test_size=internal_test_set_size, random_state=shuffle_seed, shuffle=shuffle_once
        )
        # As with MONKs, return raw numpy arrays for allowing different model selection strategies
        return x_dev, y_dev, x_test, y_test, cup_test_set_data
    else:
        # In this case there is no test data
        return x, y, None, None, cup_test_set_data


__all__ = [
    'read_monk',
    'read_cup',
]
