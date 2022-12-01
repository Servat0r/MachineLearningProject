# Callbacks for monitoring optimizers etc.
from __future__ import annotations
import json
from .base import *


class OptimizerMonitor(Callback):
    """
    Monitors optimizer state, i.e. fills a list with dict containing
    optimizer states and (optionally) logs them into a JSON file.
    """
    def __init__(self, target_list: list, target_fpath: str = None, overwrite=True):
        self.target_list = target_list
        self.target_fpath = target_fpath
        self.target_fp = None
        self.overwrite = overwrite

    def open(self):
        self.close()
        if self.target_fpath is not None:
            self.target_fp = open(self.target_fpath, 'w') if self.overwrite else open(self.target_fpath, 'a')

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
        optim_log_dict = model.optimizer.to_log_dict()
        msg = {
            'epoch': epoch,
            'state': optim_log_dict,
        }
        self.target_list.append(msg)

    def after_training_cycle(self, model, logs=None):
        if self.target_fp is not None:
            json.dump(self.target_list, fp=self.target_fp)
            self.close()


__all__ = [
    'OptimizerMonitor',
]
