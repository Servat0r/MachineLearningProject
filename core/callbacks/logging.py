# CSV Logger implemented as a callback
from __future__ import annotations
import os
from .base import *
from core.utils.types import *


class BaseCSVLogger(Callback):

    __float_types = {float, np.float, np.float_, np.float32, np.float64}  # Set will eliminate aliases

    def is_float(self, val):
        return any([isinstance(val, tp) for tp in self.__float_types])

    def fmt_float(self, val):
        tp = type(val)
        if isinstance(val, np.ndarray):
            val = val.item()
        return tp(round(val, self.round_val)) if self.round_val is not None else val

    def fmt_values(self, values):
        result = []
        for val in values:
            if self.is_float(val):
                val = self.fmt_float(val)
            result.append(val)
        return result

    def __init__(self, fpath: str = 'log.csv', overwrite=True, sep=',', round_val: int = None):
        self.fpath = fpath
        self.fp = None
        self.overwrite = overwrite
        self.sep = sep
        self.round_val = round_val

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

    def __init__(self, fpath: str = 'train_log.csv', overwrite=True, sep=',', include_mb=False, round_val: int = None):
        super(TrainingCSVLogger, self).__init__(fpath, overwrite, sep, round_val=round_val)
        self.include_mb = include_mb

    def before_training_cycle(self, model, logs=None):
        if logs is None:
            raise ValueError(f"Cannot log with no metrics!")
        self.open()
        if self.include_mb:
            print('epoch', 'is_mb', 'minibatch', *logs.keys(), sep=self.sep, file=self.fp, end='')
        else:
            print('epoch', *logs.keys(), sep=self.sep, file=self.fp, end='')
        # If logger is serialized after this method call, it will not print headers again
        self.overwrite = False

    def after_training_cycle(self, model, logs=None):
        self.close()
        print(f'CSV logging for training cycle ended')

    def before_training_epoch(self, model, epoch, logs=None):
        if not self.include_mb:
            print(file=self.fp, end=os.linesep)

    def before_training_batch(self, model, epoch, batch, logs=None):
        if self.include_mb:
            print(file=self.fp, end=os.linesep)

    def after_training_epoch(self, model, epoch, logs=None):
        values = self.fmt_values(logs.values())
        if self.include_mb:
            print(epoch, 0, 0, *values, sep=self.sep, file=self.fp, end=',')
        else:
            print(epoch, *values, sep=self.sep, file=self.fp, end=',')

    def after_training_batch(self, model, epoch, batch, logs=None):
        if self.include_mb:
            values = self.fmt_values(logs.values())
            print(epoch, 1, batch, values, sep=self.sep, file=self.fp, end='')

    def after_evaluate(self, model, epoch=None, logs=None):
        values = self.fmt_values(logs.values())
        print(*values, sep=self.sep, file=self.fp, end='')


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
