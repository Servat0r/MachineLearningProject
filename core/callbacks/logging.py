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

    def __init__(
            self, train_dirpath: str = '.', train_fpath: str = 'train_log.csv',
            overwrite=True, sep=',', include_mb=False, round_val: int = None,
    ):
        epoch_train_fpath = os.path.join(train_dirpath, train_fpath)
        mb_train_fpath = os.path.join(train_dirpath, f'minibatch_{train_fpath}')
        super(TrainingCSVLogger, self).__init__(epoch_train_fpath, overwrite, sep, round_val=round_val)
        self.include_mb = include_mb
        self.mb_fpath = mb_train_fpath
        self.mb_fp = None
        self.include_val = False

    def open(self):
        super(TrainingCSVLogger, self).open()
        if self.include_mb:
            self.mb_fp = open(self.mb_fpath, 'w') if self.overwrite else open(self.mb_fpath, 'a')

    def close(self):
        super(TrainingCSVLogger, self).close()
        if self.mb_fp is not None:
            self.mb_fp.close()
            self.mb_fp = None

    def before_training_cycle(self, model, logs=None):
        if logs is None:
            raise ValueError(f"Cannot log with no metrics!")
        train, val = logs.get('training', None), logs.get('validation', None)
        if (train is None) or (val is None):
            raise ValueError(f"Cannot log without train or validation keys!")
        self.open()
        self.include_val = len(val) > 0  # At least one validation metric specified in initial logs
        print('epoch', *train.keys(), *val.keys(), sep=self.sep, file=self.fp, flush=True)
        if self.include_mb:
            print('epoch', 'minibatch', *train.keys(), sep=self.sep, file=self.mb_fp, flush=True)
        # If logger is serialized after this method call, it will not print headers again
        self.overwrite = False

    def after_training_cycle(self, model, logs=None):
        self.close()
        print(f'CSV logging for training cycle ended', flush=True)

    def after_training_epoch(self, model, epoch, logs=None):
        values = self.fmt_values(logs.values())
        if self.include_val:
            print(epoch, *values, sep=self.sep, file=self.fp, end=',', flush=True)
        else:
            print(epoch, *values, sep=self.sep, file=self.fp, flush=True)

    def after_training_batch(self, model, epoch, batch, logs=None):
        if self.include_mb:
            values = self.fmt_values(logs.values())
            print(epoch, batch, *values, sep=self.sep, file=self.mb_fp, flush=True)

    def after_evaluate(self, model, epoch=None, logs=None):
        if self.include_val:
            values = self.fmt_values(logs.values())
            print(*values, sep=self.sep, file=self.fp, flush=True)


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


class InteractiveLogger(Callback):
    """
    Interactive logger to stdout.
    """
    def __init__(self):
        super(InteractiveLogger, self).__init__()

    def before_training_cycle(self, model, logs=None):
        print('InteractiveLogger started')

    def after_training_epoch(self, model, epoch, logs=None):
        logstr = ', '.join([f'{k} = {v}' for k, v in logs.items()])
        print(f'After training epoch {epoch}: [{logstr}]')

    def after_evaluate(self, model, epoch=None, logs=None):
        logstr = ', '.join([f'{k} = {v}' for k, v in logs.items()])
        print(f'After evaluate {epoch}: [{logstr}]')

    def after_training_cycle(self, model, logs=None):
        print(f'Training cycle ended')


__all__ = [
    'TrainingCSVLogger',
    'TestCSVLogger',
    'InteractiveLogger',
]
