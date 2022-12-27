from __future__ import annotations
import math
from core.utils import *
from sklearn.model_selection import train_test_split


class Validator:
    """
    Base class for validation approaches.
    """
    def split(self, inputs: np.ndarray, targets: np.ndarray, shuffle=True,
              random_state=None) -> Generator[tuple[np.ndarray, np.ndarray]]:
        """
        Returns a generator that is used to cycle over the training data
        for generating training-validation sets from a common development
        one each time.
        :param inputs: Input data (development set).
        :param targets: Input targets (development set).
        :param shuffle: If True, shuffles the data according to an internal
        policy (e.g. by using train_test_split for Holdout). Defaults to True.
        :param random_state: If an integer, uses it as seed for shuffling.
        Defaults to None.
        """
        pass


class KFold(Validator):
    """
    K-fold cross validation schema.
    """

    def __init__(self, number_of_folds: int):
        self.number_of_folds = number_of_folds

    def split(self, inputs: np.ndarray, targets: np.ndarray, shuffle=True, random_state=None) \
            -> Generator[tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]]:
        if shuffle:  # We want to actually permute the dataset before partitioning
            # Create random number generator
            rng = np.random.default_rng(random_state)
            permuted = rng.permutation(len(inputs), axis=0)  # Permutation of indexes
        else:
            permuted = np.arange(len(inputs))
        for i in range(self.number_of_folds):
            start_index, end_index = self.fold_indexes(inputs, i)

            validation_inputs = inputs[permuted[start_index:end_index]]
            validation_targets = targets[permuted[start_index:end_index]]

            train_inputs = np.concatenate((
                inputs[permuted[:start_index]], inputs[permuted[end_index:]],
            ), axis=0)
            train_targets = np.concatenate((
                targets[permuted[:start_index]], targets[permuted[end_index:]]
            ), axis=0)

            yield (train_inputs, train_targets), (validation_inputs, validation_targets)

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


class Holdout(Validator):
    """
    Holdout validation schema.
    """

    def split(
            self, inputs: np.ndarray, targets: np.ndarray, shuffle=True,
            random_state=None, validation_split_percentage=0.0, stratify=None
    ) -> Generator[tuple[np.ndarray, np.ndarray]]:
        if validation_split_percentage:  # != 0 AND != None
            train_x, val_x, train_y, val_y = train_test_split(
                inputs, targets, test_size=validation_split_percentage, shuffle=shuffle,
                random_state=random_state, stratify=stratify
            )
        else:
            train_x, val_x, train_y, val_y = inputs, None, targets, None
        t = (val_x, val_y) if validation_split_percentage else None
        yield [(train_x, train_y), t]


__all__ = ['Validator', 'Holdout', 'KFold']


if __name__ == '__main__':
    X = np.arange(200).reshape((100, 2))
    y = np.arange(100)
    num_folds = 5
    k_fold = KFold(num_folds)
    for train_data, eval_data in k_fold.split(X, y, random_state=0, shuffle=False):
        train_inputs, train_targets = train_data
        eval_inputs, eval_targets = eval_data
        print('Train:', train_targets)
        print('Eval:', eval_targets)
