# Optimizers
from __future__ import annotations
from ..utils import *
from .schedulers import *
from .layers import Dense, Linear, Layer


# Base class
class Optimizer:

    def __init__(self):
        self.iterations = 0
        self.epoch = 0

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_update(self):
        """
        Callback before any parameters updates.
        """
        pass

    def update(self, layers):
        """
        Main method for updating parameters.
        """
        self.before_update()
        self.update_body(layers)
        self.update_regularization_values(layers)
        self.after_update()

    def after_update(self):
        """
        Callback after any parameters updates.
        """
        pass

    def update_regularization_values(self, layers):
        """
        After having updated "standard delta values", add regularization
        (penalization) term to layers' weights.
        """
        pass

    def update_body(self, layers):
        """
        To be executed at the heart of update(), between before_update() and after_update().
        """
        pass

    def to_log_dict(self) -> TDesc:
        pass

    def reset(self):
        self.epoch = 0
        self.iterations = 0


# SGD Optimizer with optional momentum (todo add Nesterov momentum?)
class SGD(Optimizer):

    def __init__(self, lr=0.1, lr_decay_scheduler: Scheduler = None, momentum=0.):
        """
        :param lr: Initial learning rate.
        :param lr_decay_scheduler: A Scheduler (function) of the form
        (iteration, current_value) -> next_value to be used for
        learning rate decay.
        """
        super(SGD, self).__init__()
        self.initial_learning_rate = lr
        self.current_learning_rate = lr
        self.lr_decay_scheduler = lr_decay_scheduler
        self.momentum = momentum

    def __eq__(self, other):
        if not isinstance(other, SGD):
            return False
        return all([
            self.initial_learning_rate == other.initial_learning_rate,
            self.current_learning_rate == other.current_learning_rate,
            self.lr_decay_scheduler == other.lr_decay_scheduler,
            self.momentum == other.momentum,
        ])

    def before_epoch(self):
        if self.lr_decay_scheduler is not None:
            self.current_learning_rate = self.lr_decay_scheduler(self.epoch, self.current_learning_rate)

    def after_epoch(self):
        self.iterations = 0
        self.epoch += 1

    def after_update(self):
        self.iterations += 1

    def update_regularization_values(self, layers: Layer | Iterable[Layer]):
        if isinstance(layers, Dense):
            self.update_regularization_values(layers.linear)
        elif isinstance(layers, Layer):
            if isinstance(layers, Linear) and layers.is_trainable():
                if layers.weights_regularizer is not None:
                    layers.weights -= layers.weights_regularization_updates
                if layers.biases_regularizer is not None:
                    layers.biases -= layers.biases_regularization_updates
        elif isinstance(layers, Iterable):
            for layer in layers:
                self.update_regularization_values(layer)
        else:
            raise TypeError(f"Invalid type {type(layers)}: allowed ones are {Layer} or {Iterable[Layer]}")

    def update_body(self, layers: Layer | Iterable):
        if isinstance(layers, Dense):
            self.update_body(layers.linear)
        elif isinstance(layers, Layer):
            if isinstance(layers, Linear) and layers.is_trainable():
                if self.momentum:
                    self.__update_body_momentum(layers)
                else:
                    self.__update_body_no_momentum(layers)
        elif isinstance(layers, Iterable):
            for layer in layers:
                self.update_body(layer)
        else:
            raise TypeError(f"Invalid type {type(layers).__name__}: allowed ones are {Layer} or {Iterable[Layer]}")

    def __update_body_momentum(self, layer):
        if hasattr(layer, 'weight_momentums'):
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.delta_weights
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.delta_biases
            layer.bias_momentums = bias_updates

            layer.weights += weight_updates
            layer.biases += bias_updates

    def __update_body_no_momentum(self, layer):
        weight_updates = - self.current_learning_rate * layer.delta_weights
        bias_updates = - self.current_learning_rate * layer.delta_biases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def to_log_dict(self) -> TDesc:
        return {
            'lr': self.current_learning_rate,
        }


__all__ = [
    'Optimizer',
    'SGD',
]
