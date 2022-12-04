# History objects (similar to Keras ones) for logging the history of the training of a model
# while Model.train() is ongoing
from __future__ import annotations
from core.utils.types import *
from core.callbacks import Callback


class History(Callback):

    def __init__(self, max_epochs: int, dtype=np.float64):
        self.logbook = {
            'loss': np.zeros(max_epochs, dtype=dtype),
            'Val_loss': np.zeros(max_epochs, dtype=dtype),
        }
        self.epoch = 0
        self.max_epochs = max_epochs

    def __len__(self):
        # Conveniently define length as the number of epochs for which there exists valid data in the history
        return self.epoch

    def __eq__(self, other):
        if not isinstance(other, History):
            return False
        check = all([
            self.epoch == other.epoch, self.max_epochs == other.max_epochs, len(self.logbook) == len(other.logbook)
        ])
        if not check:
            return False
        for (k1, v1), (k2, v2) in zip(self.logbook.items(), other.logbook.items()):
            check = (k1 == k2) and np.equal(v1, v2).all()
            if not check:
                return False
        return True

    def keys(self):
        return self.logbook.keys()

    def items(self):
        return self.logbook.items()

    def values(self):
        return self.logbook.values()

    def __getitem__(self, item):
        return self.logbook[item]

    def get_epoch_data(self, epoch: int):
        if epoch >= self.max_epochs:
            raise IndexError(f"Invalid epoch number {epoch}: maximum is {self.max_epochs - 1}")
        elif epoch > self.epoch:
            raise IndexError(f"Invalid epoch number {epoch}: last terminated epoch is {self.epoch - 1}")
        return {k: v[epoch] for k, v in self.logbook.items()}

    def before_training_cycle(self, model, logs=None):
        metrics = model.metrics
        if metrics is not None:
            for metric in metrics:
                self.logbook[metric.get_name()] = np.zeros(self.max_epochs, dtype=metric.dtype)
        model.history = self

    def after_training_epoch(self, model, epoch, logs=None):
        self.update_values(model, epoch, logs)

    def after_evaluate(self, model, epoch=None, logs=None):
        self.update_values(model, epoch, logs)

    def after_training_cycle(self, model, logs=None):
        model.history = self

    def update_values(self, model, epoch, logs=None):
        if logs is not None:
            for key, val in logs.items():
                key_arr = self.logbook.get(key, None)
                if key_arr is None:
                    raise RuntimeError(f"Unknown metric {key} (epoch = {epoch})")
                if val is None:
                    raise RuntimeError(f"None value passed for {key} after epoch {epoch} (expected numerical value)")
                if isinstance(val, np.ndarray):
                    val = val.item()
                key_arr[epoch] = val
            self.epoch = epoch + 1
            model.history = self


__all__ = [
    'History',
]
