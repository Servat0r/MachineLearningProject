# Callbacks for monitoring optimizers etc.
from __future__ import annotations
import json
import copy
from core.utils.types import *
from .base import *


class OptimizerMonitor(Callback):
    """
    Monitors optimizer state, i.e. fills a list with dict containing
    optimizer states and (optionally) logs them into a JSON file.
    """
    def __init__(self, target_list: list, target_file_path: str = None, overwrite=True):
        self.target_list = target_list
        self.target_file_path = target_file_path
        self.target_fp = None
        self.overwrite = overwrite

    def open(self):
        self.close()
        if self.target_file_path is not None:
            self.target_fp = open(self.target_file_path, 'w') if self.overwrite else open(self.target_file_path, 'a')

    def close(self):
        if self.target_fp is not None:
            self.target_fp.close()
            self.target_fp = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('target_fp')
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.target_fp = None

    def before_training_cycle(self, model, logs=None):
        self.open()

    def after_training_epoch(self, model, epoch, logs=None):
        optimizer_log_dict = model.optimizer.to_log_dict()
        msg = {
            'epoch': epoch,
            'state': optimizer_log_dict,
        }
        self.target_list.append(msg)

    def after_training_cycle(self, model, logs=None):
        if self.target_fp is not None:
            json.dump(self.target_list, fp=self.target_fp)
            self.close()


class ModelMonitor(Callback):
    """
    A callback similar to EarlyStopping in the sense that maintains trace of the best model at each time,
    accordingly to a given metric.
    """
    def __init__(self, monitor='Val_loss', mode='min', return_best_result=False):
        self.monitor = monitor
        self.mode = mode
        self.is_validation_metric = None   # Whether metric is for training or validation
        self.return_best_result = return_best_result
        self.best_metric_value = None  # Best value for monitored metric (independent from min_delta)
        self.best_parameters = None
        self.best_epoch = None

    def before_training_cycle(self, model, logs=None):
        if not self.monitor.startswith('Val_'):
            self.is_validation_metric = False
        elif self.monitor == 'Val_loss':
            self.is_validation_metric = True
        else:
            for val_metric in model.validation_metrics:
                if val_metric.name == self.monitor:
                    self.is_validation_metric = True

    def after_training_epoch(self, model, epoch, logs=None):
        if not self.is_validation_metric:  # training metric is being monitored
            self._main(model, epoch, logs)

    def after_evaluate(self, model, epoch=None, logs=None):
        if self.is_validation_metric:  # validation metric is being monitored
            self._main(model, epoch, logs)

    def _main(self, model, epoch, logs=None):
        target_metric = logs.get(self.monitor, None)
        if self.return_best_result and ((self.best_parameters is None) or self._is_better(target_metric)):
            self.best_parameters = model.get_parameters(copy=True)
            self.best_epoch = epoch
            self.best_metric_value = target_metric

    def _is_better(self, target_metric):
        """
        Defines if a metric value is better (< or >) than the *best* one (regardless of min_delta).
        """
        if self.mode == 'min':
            return target_metric < self.best_metric_value
        else:
            return target_metric > self.best_metric_value

    # todo we can set parameters with after_training_cycle() directly
    #  into the original model instead of copying every time
    def get_best_parameters(self):
        return self.best_parameters

    def get_best_epoch(self):
        return self.best_epoch

    def get_best_value(self):
        return self.best_metric_value


class TestSetMonitor(Callback):
    """
    A callback that monitors test set data during training.
    """
    def __init__(
            self, test_set_data: np.ndarray, test_set_targets: np.ndarray,
            metrics, max_epochs: int, dtype=np.float32
    ):
        self.logbook = {metric.get_name(): np.zeros(max_epochs, dtype=dtype) for metric in metrics}
        self.logbook['loss'] = np.zeros(max_epochs, dtype=dtype)
        self.metrics = copy.deepcopy(metrics)
        self.test_set_data = test_set_data
        self.test_set_targets = test_set_targets
        self.epoch = 0
        self.max_epochs = max_epochs

    def __len__(self):
        return self.epoch

    def after_training_epoch(self, model, epoch, logs=None):
        model.set_to_eval(detach_history=False)
        test_set_net_output = model.forward(self.test_set_data)
        # Calculate metrics values
        for metric in self.metrics:
            metric.before_batch()
            self.logbook[metric.get_name()][epoch] = metric.update(test_set_net_output, self.test_set_targets)
            metric.after_batch()
        # Calculate loss value
        data_loss, regularization_loss = model.loss.forward(
            test_set_net_output, self.test_set_targets, layers=model.layers
        )
        self.logbook['loss'][epoch] = data_loss + regularization_loss
        # Update current epoch
        self.epoch = epoch + 1

    def __getitem__(self, item):
        return self.logbook[item]


__all__ = [
    'OptimizerMonitor',
    'ModelMonitor',
    'TestSetMonitor',
]
