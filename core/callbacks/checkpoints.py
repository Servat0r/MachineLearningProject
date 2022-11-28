# Implementation of Model checkpoints callbacks
from __future__ import annotations
from .base import *


class ModelCheckpoint(Callback):
    """
    Implements a callback that saves the current model in eval mode
    after a certain (predefined) number of epochs (defaults to 1)
    and at the end of the training cycle.
    """
    def __init__(self, fpath='model_checkpoint.model', save_every=1, save_history=True):
        self.fpath = fpath
        self.save_every = save_every
        self.save_history = save_history

    def __save_model(self, model):
        history = None
        if not self.save_history:
            history = model.set_to_eval(detach_history=True)
        # Save model in specified path
        model.save(self.fpath, include_compile_objs=False, include_history=self.save_history)
        if not self.save_history:
            model.history = history

    def after_training_epoch(self, model, epoch, logs=None):
        if epoch % self.save_every == 0:
            self.__save_model(model)

    def after_training_cycle(self, model, logs=None):
        self.__save_model(model)


class ModelBackup(Callback):
    """
    Callback that saves current model state, INCLUDING optimizer, loss,
    metrics, weights updates etc. for suspend and resume training.
    """
    def __init__(self, fpath='model_backup.model', save_every=1):
        self.fpath = fpath
        self.save_every = save_every

    def __save_model(self, model):
        is_training = model.is_training()
        model.set_to_train()
        # For saving also updating status (todo modify model for saving momentums)
        model.save(self.fpath, include_compile_objs=True, include_history=True)
        if not is_training:
            model.set_to_eval(detach_history=False)

    def after_training_epoch(self, model, epoch, logs=None):
        if epoch % self.save_every == 0:
            self.__save_model(model)

    def after_training_cycle(self, model, logs=None):
        self.__save_model(model)


__all__ = [
    'ModelCheckpoint',
    'ModelBackup',
]
