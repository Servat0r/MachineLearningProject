# Optimizers
from __future__ import annotations
from ..utils import *
from .schedulers import *
from .layers import SequentialLayer, FullyConnectedLayer


# Base class
class Optimizer:

    def __init__(self):
        self.iterations = 0

    @abstractmethod
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
        self.update_reg_values(layers)
        self.after_update()

    @abstractmethod
    def after_update(self):
        """
        Callback after any parameters updates.
        """
        pass

    @abstractmethod
    def update_reg_values(self, layers):
        """
        Subroutine for handling regularization terms in updating.
        """

    @abstractmethod
    def update_body(self, layers):
        """
        To be executed at the heart of update(), between before_update() and after_update().
        """
        pass

    def get_num_iterations(self) -> int:
        return self.iterations

    def reset_iterations(self, dump=True) -> int | None:
        iters = self.iterations if dump else None
        self.iterations = 0
        return iters


# SGD Optimizer with optional momentum (todo add Nesterov momentum?)
class SGD(Optimizer):

    def __init__(self, lr=0.1, lr_decay_scheduler: Scheduler = None, momentum=0.):
        super(SGD, self).__init__()
        self.lr = lr
        self.current_lr = lr
        self.lr_decay_scheduler = lr_decay_scheduler
        self.momentum = momentum

    def before_update(self):
        if self.lr_decay_scheduler is not None:
            self.current_lr = self.lr_decay_scheduler(self.iterations, self.current_lr)

    @staticmethod
    def __update_l1_reg(layer):
        if hasattr(layer, 'l1_regularizer') and layer.l1_regularizer != 0.:
            weight_ones = np.sign(layer.weights)
            bias_ones = np.sign(layer.biases)

            layer.weights -= layer.l1_regularizer * weight_ones
            layer.biases -= layer.l1_regularizer * bias_ones

    @staticmethod
    def __update_l2_reg(layer):
        if hasattr(layer, 'l2_regularizer') and layer.l2_regularizer != 0.:
            layer.weights -= layer.l2_regularizer * layer.weights
            layer.biases -= layer.l2_regularizer * layer.biases

    # todo fixme L1 and L2 regularizers are applied AFTER subtracting dweights and dbiases! Is this correct??
    def update_reg_values(self, layers: SequentialLayer | Iterable):
        if isinstance(layers, SequentialLayer):
            self.update_reg_values(layers.layers)
        elif isinstance(layers, FullyConnectedLayer):
            self.update_reg_values({layers.linear})
        elif isinstance(layers, Iterable):
            for layer in layers:
                if isinstance(layer, SequentialLayer):
                    self.update_reg_values(layer.layers)
                elif isinstance(layer, FullyConnectedLayer):
                    self.update_reg_values({layer.linear})
                elif layer.is_parametrized():
                    self.__update_l1_reg(layer)
                    self.__update_l2_reg(layer)
        else:
            raise TypeError(f"Invalid type {type(layers).__name__}: allowed ones are "
                            f"{SequentialLayer}, {FullyConnectedLayer} or {Iterable}")

    def __update_body_momentum(self, layer):
        if hasattr(layer, 'weight_momentums'):
            weight_updates = self.momentum * layer.weight_momentums - self.current_lr * layer.dweights
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_lr * layer.dbiases
            layer.bias_momentums = bias_updates

            layer.weights += weight_updates
            layer.biases += bias_updates

    def __update_body_no_momentum(self, layer):
        weight_updates = - self.current_lr * layer.dweights
        bias_updates = - self.current_lr * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def update_body(self, layers: SequentialLayer | Iterable):
        if isinstance(layers, SequentialLayer):
            self.update_body(layers.layers)
        elif isinstance(layers, FullyConnectedLayer):
            self.update_body({layers.linear})
        elif isinstance(layers, Iterable):
            for layer in layers:
                if isinstance(layer, SequentialLayer):
                    self.update_body(layer.layers)
                elif isinstance(layer, FullyConnectedLayer):
                    self.update_body({layer.linear})
                elif layer.is_parametrized():
                    if self.momentum:
                        self.__update_body_momentum(layer)
                    else:
                        self.__update_body_no_momentum(layer)
        else:
            raise TypeError(f"Invalid type {type(layers).__name__}: allowed ones are"
                            f"{SequentialLayer}, {FullyConnectedLayer} or {Iterable}")

    def after_update(self):
        self.iterations += 1


__all__ = [
    'Optimizer',
    'SGD',
]
