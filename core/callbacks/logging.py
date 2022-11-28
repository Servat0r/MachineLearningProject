# CSV Logger implemented as a callback
from __future__ import annotations
from core.utils.types import *
from .base import *


class BaseCSVLogger(Callback):

    def __init__(self, fpath: str = 'log.csv', overwrite=True, sep=','):
        self.fpath = fpath
        self.fp = None
        self.overwrite = overwrite
        self.sep = sep

    def open(self):
        self.close()
        self.fp = open(self.fpath, 'w') if self.overwrite else open(self.fpath, 'a')

    def close(self):
        if self.fp is not None:
            self.fp.close()
            self.fp = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('fp')
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.fp = None


class TrainingCSVLogger(BaseCSVLogger):

    def __init__(self, fpath: str = 'train_log.csv', overwrite=True, sep=',', include_mb=False):
        super(TrainingCSVLogger, self).__init__(fpath, overwrite, sep)
        self.include_mb = include_mb

    def before_training_cycle(self, model, logs=None):
        if logs is None:
            raise ValueError(f"Cannot log with no metrics!")
        self.open()
        if self.include_mb:
            print('epoch', 'is_mb', 'minibatch', *logs.keys(), sep=self.sep, file=self.fp)
        else:
            print('epoch', *logs.keys(), sep=self.sep, file=self.fp)
        # If logger is serialized after this method call, it will not print headers again
        self.overwrite = False

    def after_training_cycle(self, model, logs=None):
        self.close()
        print(f'CSV logging for training cycle ended')

    def after_training_epoch(self, model, epoch, logs=None):
        if self.include_mb:
            print(epoch, 0, 0, *logs.values(), sep=self.sep, file=self.fp)
        else:
            print(epoch, *logs.values(), sep=self.sep, file=self.fp)

    def after_training_batch(self, model, epoch, batch, logs=None):
        if self.include_mb:
            print(epoch, 1, batch, *logs.values(), sep=self.sep, file=self.fp)


class TestCSVLogger(BaseCSVLogger):

    def __init__(self, fpath: str = 'test_log.csv', overwrite=True, sep=','):
        super(TestCSVLogger, self).__init__(fpath, overwrite, sep)
        self.example = 0  # Next example num

    def before_test_cycle(self, model, logs=None):
        if logs is None:
            raise ValueError(f"Cannot log with no metrics!")
        self.open()
        print('example', *logs.keys(), sep=self.sep, file=self.fp)
        # If logger is serialized after this method call, it will not print headers again
        self.overwrite = False

    def after_test_cycle(self, model, logs=None):
        self.close()
        print('CSV Logging for test cycle has ended')

    def after_test_batch(self, model, logs=None):
        for example, values in enumerate(zip(*logs.values())):
            print(example + self.example, *values, sep=self.sep, file=self.fp)
            self.example += 1


__all__ = [
    'TrainingCSVLogger',
    'TestCSVLogger',
]
