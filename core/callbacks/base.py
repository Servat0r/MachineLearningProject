# General callbacks API (metrics are a subset of callbacks)
from __future__ import annotations


class Callback:
    """
    A generic callback that can be plugged in the training/validation cycle
    of a model to customize behaviour.

    Callbacks are defined in terms of the before/after_training_cycle/epoch/batch
    and before/after_evaluate methods, which accept the current Model object and
    a series of logs that are the currently (i.e. at epoch/minibatch level) values
    computed by loss and metrics.

    An example of usage:
    # suppose model, train_dataloader and eval_dataloader are defined variables
    callbacks = [InteractiveLogger(), EarlyStopping()]
    model.train(train_dataloader, eval_dataloader, callbacks=callbacks)

    The above code will print to screen loss and metric values for each epoch
    (InteractiveLogger) and use default values for EarlyStopping during training.
    """

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


__all__ = [
    'Callback',
]
