# General callbacks API (metrics are a subset of callbacks)
from __future__ import annotations
from core.utils.types import *


class Callback(Callable):

    def before_training_cycle(self, model, logs=None):
        pass

    def before_training_epoch(self, model, epoch, logs=None):
        pass

    def before_training_batch(self, model, epoch, batch, logs=None):
        pass

    def after_training_batch(self, model, epoch, batch, logs=None):
        pass

    def after_training_epoch(self, model, epoch, logs=None):
        pass

    def after_training_cycle(self, model, logs=None):
        pass

    def before_evaluate(self, model, epoch=None, logs=None):
        pass

    def after_evaluate(self, model, epoch=None, logs=None):
        pass

    def before_test_cycle(self, model, logs=None):
        pass

    def before_test_batch(self, model, logs=None):
        pass

    def after_test_batch(self, model, logs=None):
        pass

    def after_test_cycle(self, model, logs=None):
        pass

    def before_predict(self, model, logs=None):
        pass

    def after_predict(self, model, logs=None):
        pass


__all__ = [
    'Callback',
]
