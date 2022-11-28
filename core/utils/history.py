# History objects (similar to Keras ones) for logging the history of the training of a model
# while Model.train() is ongoing
from __future__ import annotations

import numpy as np

from core.utils.types import *
import core.utils as cu


class History:

    def __init__(self, n_epochs: int, n_batches: int, loss, metrics=None):
        self.logbook = {
            'loss': np.empty(n_epochs, dtype=loss.dtype)
        }
        self.batch_vals = {
            'loss': np.empty(n_batches, dtype=loss.dtype)
        }
        if metrics is not None:
            for metric in metrics:
                self.logbook[metric.get_name()] = np.empty(n_epochs, dtype=metric.dtype)
                self.batch_vals[metric.get_name()] = np.empty(n_batches, dtype=metric.dtype)
        self.epoch = 0
        self.batch = 0
        self.n_epochs = n_epochs
        self.n_batches = n_batches

    def keys(self):
        return self.logbook.keys()

    def items(self):
        return self.logbook.items()

    def values(self):
        return self.logbook.values()

    def __getitem__(self, item):
        return self.logbook[item]

    def epoch_data(self, epoch: int):
        if epoch >= self.n_epochs:
            raise IndexError(f"Invalid epoch number {epoch}: maximum is {self.n_epochs - 1}")
        elif epoch > self.epoch:
            raise IndexError(f"Invalid epoch number {epoch}: last terminated epoch is {self.epoch - 1}")
        return {k: v[epoch] for k, v in self.logbook.items()}

    def before_train_cycle(self):
        pass

    def before_train_epoch(self):
        pass

    def before_train_batch(self):
        pass

    def after_train_batch(self):
        pass

    def after_train_epoch(self):
        pass

    def after_train_cycle(self):
        pass

