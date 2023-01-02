# Early Stopping callback
from __future__ import annotations
import copy
from .base import *


class EarlyStopping(Callback):
    """
    EarlyStopping regularization strategy.
    """

    def __init__(self, monitor='Val_loss', min_delta=0, patience=0, mode='min', return_best_result=False):
        """
        :param monitor: Which metric to monitor.
        :param min_delta: Minimum difference from previous metric value and current one to consider new value
        an improvement (this prevents to fit when improvements are too much small). Defaults to 0.
        :param patience: Maximum number of epochs that can elapse without an improvement (as defined above)
        before this callback stops model training. Defaults to 0.
        :param mode: 'min' for a metric that shall be minimized (e.g. MEE), 'max' if metric shall be maximized
        (e.g. accuracy).
        :param return_best_result: If True, attributes 'best_metric_value', 'best_model', 'best_epoch' will be
        filled during training loop and available at the end. Defaults to False.
        """
        self.monitor = monitor
        self.min_delta = abs(min_delta)
        self.patience = patience
        self.mode = mode
        self.last_value_recorded = None  # Last value recorded for the given metric
        self.elapsed_patience_epochs = 0  # How many epochs have elapsed without improvement
        self.is_validation_metric = None   # Whether metric is for training or validation
        self.return_best_result = return_best_result
        self.best_metric_value = None  # Best value for monitored metric (independent from min_delta)
        self.best_model = None
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
        if target_metric is not None:
            if self.last_value_recorded is None:
                self.last_value_recorded = target_metric
            if self.return_best_result and ((self.best_model is None) or self._is_better(target_metric)):
                self.best_model = copy.deepcopy(model)
                self.best_epoch = epoch
                self.best_metric_value = target_metric
            if self._is_improving(target_metric):
                self.last_value_recorded = target_metric
                self.elapsed_patience_epochs = 0
            else:
                self.elapsed_patience_epochs += 1
                if self.elapsed_patience_epochs > self.patience:
                    model.stop_training = True
                    print(f'{type(self).__name__} stopped training at epoch {epoch}')

    def _is_better(self, target_metric):
        """
        Defines if a metric value is better (< or >) than the *best* one (regardless of min_delta).
        """
        if self.mode == 'min':
            return target_metric < self.best_metric_value
        else:
            return target_metric > self.best_metric_value

    def _is_improving(self, target_metric):
        """
        Defines if a metric value is better (< or >) than the *last* one, including min_delta.
        """
        if self.mode == 'min':
            return target_metric + self.min_delta < self.last_value_recorded
        else:
            return target_metric > self.last_value_recorded + self.min_delta

    def get_best(self):
        return self.best_model

    def get_best_epoch(self):
        return self.best_epoch

    def get_best_value(self):
        return self.best_metric_value


__all__ = [
    'EarlyStopping',
]
