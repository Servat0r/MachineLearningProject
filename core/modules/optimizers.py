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
        result = self.update_body(layers)
        self.after_update()
        return result

    @abstractmethod
    def after_update(self):
        """
        Callback after any parameters updates.
        """
        pass

    @abstractmethod
    def update_reg_values(self, parameter, w_vals: np.ndarray, b_vals: np.ndarray):
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

    def update_reg_values(self, parameter, w_vals: np.ndarray, b_vals: np.ndarray):
        """
        Subroutine for handling regularization terms in updating.
        """
        for reg_name, reg_updates in parameter.regularizer_updates.items():
            regw_updates, regb_updates = reg_updates.get('weights'), reg_updates.get('biases')
            if regw_updates is not None:
                w_vals += regw_updates  # self.apply_reduction(regw_updates) todo sure? we are ignoring batch size!
            if regb_updates is not None:
                b_vals += regb_updates  # self.apply_reduction(regb_updates) todo same as above
        return w_vals, b_vals   # todo necessary?

    def __update_body_momentum(self, layer):
        if isinstance(layer, SequentialLayer):
            self.update_body(layer.layers)
        elif isinstance(layer, FullyConnectedLayer):
            self.update_body({layer.linear})
        elif hasattr(layer, 'weight_momentums') and layer.is_parametrized():
            weight_updates = self.momentum * layer.weight_momentums - self.current_lr * layer.dweights
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_lr * layer.dbiases
            layer.bias_momentums = bias_updates

            layer.weights += weight_updates
            layer.biases += bias_updates

    def __update_body_no_momentum(self, layer):
        if isinstance(layer, SequentialLayer):
            self.update_body(layer.layers)
        elif isinstance(layer, FullyConnectedLayer):
            self.update_body({layer.linear})
        elif layer.is_parametrized():
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
            if self.momentum:
                for layer in layers:
                    self.__update_body_momentum(layer)
            else:
                for layer in layers:
                    self.__update_body_no_momentum(layer)
        else:
            raise TypeError(f"Invalid type {type(layers).__name__}: allowed ones are {SequentialLayer} or {Iterable}")

    def after_update(self):
        self.iterations += 1


__all__ = [
    'Optimizer',
    'SGD',
]
