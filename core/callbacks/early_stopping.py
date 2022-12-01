# Early Stopping callback
from __future__ import annotations
import copy
from .base import *


# todo add model.get_parameters(copy=True) after each epoch and restore best results at the end
class EarlyStopping(Callback):

    def __init__(self, monitor='Val_loss', min_delta=0, patience=0, mode='min', return_best=False):
        self.monitor = monitor
        self.min_delta = abs(min_delta)
        self.patience = patience
        self.mode = mode
        self.last_value = None  # Last value recorded for the given metric
        self.patience_epochs = 0  # How many epochs have elapsed without improvement
        self.is_val_metric = None   # Whether metric is for training or validation
        self.return_best = return_best
        self.best = None
        self.best_epoch = None

    def _is_improving(self, target_metric):
        if self.mode == 'min':
            return target_metric + self.min_delta < self.last_value
        else:
            return target_metric > self.last_value + self.min_delta

    def _main(self, model, epoch, logs=None):
        target_metric = logs.get(self.monitor, None)
        if (self.last_value is None) or self._is_improving(target_metric):
            self.last_value = target_metric
            self.patience_epochs = 0
            if self.return_best:
                self.best = copy.deepcopy(model)
                self.best_epoch = epoch
        else:
            self.last_value = target_metric
            self.patience_epochs += 1
            if self.patience_epochs >= self.patience:  # todo >= or > ?
                model.stop_training = True

    def before_training_cycle(self, model, logs=None):
        if not self.monitor.startswith('Val_'):
            self.is_val_metric = False
        else:
            for val_metric in model.validation_metrics:
                if val_metric.name == self.monitor:
                    self.is_val_metric = True

    def after_training_epoch(self, model, epoch, logs=None):
        if not self.is_val_metric:  # training metric is being monitored
            self._main(model, epoch, logs)

    def after_evaluate(self, model, epoch=None, logs=None):
        if self.is_val_metric:  # validation metric is being monitored
            self._main(model, epoch, logs)

    # todo we can set parameters with after_training_cycle() directly
    #  into the original model instead of copying every time
    def get_best(self):
        return self.best

    def get_best_epoch(self):
        return self.best_epoch


__all__ = [
    'EarlyStopping',
]
