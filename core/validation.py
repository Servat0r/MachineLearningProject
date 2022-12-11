from __future__ import annotations
import math
from core.utils import *
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class KFoldValidation:

    def __init__(self, number_of_folds: int, seed: int = None):
        self.number_of_folds = number_of_folds
        # Reproducibility with seed
        self.seed = seed

    def split(self, dataset: np.ndarray, targets: np.ndarray, shuffle=True) -> list[tuple[np.ndarray, np.ndarray]]:
        if shuffle:  # We want to actually permute the dataset before partitioning
            # Create random number generator
            rng = np.random.default_rng(self.seed)
            permuted = rng.permutation(len(dataset), axis=0)  # Permutation of indexes
        else:
            permuted = np.arange(len(dataset))
        dataset_split = []
        for i in range(self.number_of_folds):
            start_index, end_index = self.fold_indexes(dataset, i)
            dataset_split.append((
                dataset[permuted[start_index:end_index]],
                targets[permuted[start_index:end_index]],
            ))
        return dataset_split  # number_of_folds arrays

    def fold_indexes(self, dataset: np.ndarray, fold_num: int) -> tuple[int, int]:
        base_fold_size = math.floor(len(dataset) / self.number_of_folds)  # Base fold size (until overflowing)
        start_index = fold_num * base_fold_size
        if fold_num + 1 < self.number_of_folds:
            # If fold is not the last one, fold size will be base one
            end_index = start_index + base_fold_size
        else:
            # Otherwise, we will count up to the end of the dataset (i.e.,
            # if dataset length is NOT a multiple of base_fold_size, we
            # consider a larger/smaller fold that contains elements up to
            # the end of the dataset)
            end_index = len(dataset)
        return start_index, end_index   # length of the fold == end_index - start_index


class Holdout:

    def split(
            self, X, y, split_percentage, validation_split_percentage=0, shuffle=True, stratify=None, random_state=None
    ):
        dev_x, test_x, dev_y, test_y = train_test_split(
            X, y, test_size=split_percentage, shuffle=shuffle, random_state=random_state, stratify=stratify
        )
        if validation_split_percentage:  # != 0 AND != None
            train_x, val_x, train_y, val_y = train_test_split(
                dev_x, dev_y, test_size=validation_split_percentage, shuffle=shuffle,
                random_state=random_state, stratify=stratify
            )
        else:
            train_x, val_x, train_y, val_y = dev_x, None, dev_y, None
        return train_x, train_y, val_x, val_y, test_x, test_y


if __name__ == '__main__':
    X, y = load_breast_cancer(return_X_y=True)
    num_folds = 5
    k_fold = KFoldValidation(num_folds)
    folds1 = k_fold.split(X, y)
