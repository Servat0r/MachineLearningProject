# Implementation of Model checkpoints callbacks
from __future__ import annotations
from .base import *


class ModelCheckpoint(Callback):
    """
    Implements a callback that saves the current model in eval mode
    after a certain (predefined) number of epochs (defaults to 1)
    and at the end of the training cycle.
    """
    def __init__(self, file_path='model_checkpoint.model', save_every=1, save_history=True):
        self.file_path = file_path
        self.save_every = save_every
        self.save_history = save_history

    def after_training_epoch(self, model, epoch, logs=None):
        if (epoch % self.save_every == 0) or model.stop_training:
            self.__save_model(model)
            print(f'Saved model at {epoch} training epoch')

    def after_training_cycle(self, model, logs=None):
        self.__save_model(model)
        print(f'Saved model after training cycle')

    def __save_model(self, model):
        model.set_to_eval()
        model.save(self.file_path, include_compile_objs=False, include_history=self.save_history)


class ModelBackup(Callback):
    """
    Callback that saves current model state, INCLUDING optimizer, loss,
    metrics, weights updates etc. for suspend and resume training.
    """
    def __init__(self, file_path='model_backup.model', save_every=1):
        self.file_path = file_path
        self.save_every = save_every

    def after_training_epoch(self, model, epoch, logs=None):
        if (epoch % self.save_every == 0) or model.stop_training:
            self.__save_model(model)
            print(f'Saved model at {epoch} training epoch')

    def after_training_cycle(self, model, logs=None):
        self.__save_model(model)
        print(f'Saved model after training cycle')

    def __save_model(self, model):
        # For saving also updating status (todo modify linear layer for saving momentums)
        model.save(self.file_path, include_compile_objs=True, include_history=True, serialize_all=True)


__all__ = [
    'ModelCheckpoint',
    'ModelBackup',
]
