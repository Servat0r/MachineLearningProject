# CSV Logger implemented as a callback
from __future__ import annotations
import os
from .base import *
from core.utils.types import *


class BaseCSVLogger(Callback):
    """
    Base class (common methods) for CSV loggers.
    """

    # Below are all possible float types and are used for formatting float values when self.float_round_val != None
    __float_types = {float, np.float, np.float_, np.float32, np.float64}  # Set will eliminate aliases

    def __init__(self, file_path: str = 'log.csv', overwrite=True, separator=',', float_round_val: int = None):
        """
        :param file_path: Path of the file used for logging results.
        :param overwrite: If True, overwrites content of the logging file. Defaults to True.
        :param separator: CSV separator to use when logging.
        :param float_round_val: If not None, floating point values will be logged as rounded
        to the given number of digits for better readability of the results. Defaults to None.
        """
        self.file_path = file_path
        self.fp = None
        self.overwrite = overwrite
        self.separator = separator
        self.float_round_val = float_round_val

    def open(self):
        self.close()  # Closes any other logging file that is still open
        self.fp = open(self.file_path, 'w') if self.overwrite else open(self.file_path, 'a')

    def close(self):
        if self.fp is not None:
            self.fp.close()
            self.fp = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('fp')  # Cannot serialize an IO buffer.
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.fp = None  # Restores IO buffer as if it was closed.

    def format_values(self, values):
        """
        Shortcut for iterating over all logging value and formatting floats when necessary.
        """
        result = []
        for val in values:
            if self.is_float(val):
                val = self.format_float(val)
            result.append(val)
        return result

    def is_float(self, val):
        return any([isinstance(val, tp) for tp in self.__float_types])

    def format_float(self, val):
        """
        Formats floats according to the given round value (number of digits to maintain).
        """
        tp = type(val)
        if isinstance(val, np.ndarray):
            val = val.item()
        return tp(round(val, self.float_round_val)) if self.float_round_val is not None else val


class TrainingCSVLogger(BaseCSVLogger):
    """
    Training/Validation loops - tailored CSV logger. This logger can produce three outputs:
        - a single log file with training metrics values if self.include_minibatch_logging
            = False and no validation metrics are provided;
        - a single log file with training and validation metrics values if there are validation
            ones and self.include_minibatch_logging = False;
        - the file described above (for both cases) and another log file with all training
            metrics values at minibatch level if self.include_minibatch_logging = True.
    """

    def __init__(
            self, train_directory_path: str = '.', train_file_name: str = 'train_log.csv',
            overwrite=True, separator=',', include_minibatch_logging=False, float_round_val: int = None,
    ):
        """
        :param train_directory_path: Directory for training log file.
        :param train_file_name: Name of the training log file.
        :param overwrite: Whether to overwrite the content of the log file if existing.
        :param separator: CSV separator to use for logging.
        :param include_minibatch_logging: Whether to log also minibatch training metrics values.
        :param float_round_val: Number of digits for formatting float values.
        """
        epoch_train_fpath = os.path.join(train_directory_path, train_file_name)
        mb_train_fpath = os.path.join(train_directory_path, f'minibatch_{train_file_name}')
        super(TrainingCSVLogger, self).__init__(
            epoch_train_fpath, overwrite, separator, float_round_val=float_round_val
        )
        self.include_minibatch_logging = include_minibatch_logging
        self.minibatch_logfile_path = mb_train_fpath
        self.minibatch_fp = None
        self.include_validation_logs = False

    def open(self):
        super(TrainingCSVLogger, self).open()
        if self.include_minibatch_logging:
            self.minibatch_fp = open(self.minibatch_logfile_path, 'w') \
                if self.overwrite else open(self.minibatch_logfile_path, 'a')

    def close(self):
        super(TrainingCSVLogger, self).close()
        if self.minibatch_fp is not None:
            self.minibatch_fp.close()
            self.minibatch_fp = None

    def before_training_cycle(self, model, logs=None):
        if logs is None:
            raise ValueError(f"Cannot log with no metrics!")
        train, val = logs.get('training', None), logs.get('validation', None)
        if (train is None) or (val is None):
            raise ValueError(f"Cannot log without train or validation keys!")
        self.open()
        self.include_validation_logs = len(val) > 0  # At least one validation metric specified in initial logs
        # Print all metric names (including 'loss' and 'Val_loss')
        print('epoch', *train.keys(), *val.keys(), sep=self.separator, file=self.fp, flush=True)
        if self.include_minibatch_logging:
            print('epoch', 'minibatch', *train.keys(), sep=self.separator, file=self.minibatch_fp, flush=True)
        # If logger is serialized after this method call, it will not print headers again
        self.overwrite = False

    def after_training_cycle(self, model, logs=None):
        self.close()
        print(f'CSV logging for training cycle ended', flush=True)

    def after_training_epoch(self, model, epoch, logs=None):
        values = self.format_values(logs.values())
        if self.include_validation_logs:
            # Do not go to the next line in order to print also validation results
            print(epoch, *values, sep=self.separator, file=self.fp, end=',', flush=True)
        else:
            # Go to the next line since there won't be any validation results printout
            print(epoch, *values, sep=self.separator, file=self.fp, flush=True)

    def after_training_batch(self, model, epoch, batch, logs=None):
        if self.include_minibatch_logging:
            values = self.format_values(logs.values())
            print(epoch, batch, *values, sep=self.separator, file=self.minibatch_fp, flush=True)

    def after_evaluate(self, model, epoch=None, logs=None):
        if self.include_validation_logs:
            values = self.format_values(logs.values())
            print(*values, sep=self.separator, file=self.fp, flush=True)


class InteractiveLogger(Callback):
    """
    Interactive logger to stdout: for each epoch,
    it prints to stdout all current logs.
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
    'InteractiveLogger',
]
