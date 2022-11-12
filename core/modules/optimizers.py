# Optimizers
from __future__ import annotations
from ..utils import *
from .parameters import WeightedLayerParameters as WLParameters
from .layers import *
from .losses import *


# Base class
class Optimizer:

    # An example of exponential lr decay that can be given as argument when initializing
    @staticmethod
    def exp_lr_decay(decay: float):  # todo is 'exponential'? (check slides!)
        return lambda iters, lr: 1. / (1. + decay * iters)

    # An example of linear lr decay that can be given as argument when initializing
    @staticmethod
    def linear_lr_decay(start_lr: float, end_lr: float, max_iter: int):
        def closure(iters: int, lr: float):
            beta = iters / max_iter
            return (1. - beta) * start_lr + beta * end_lr

        return closure

    def __init__(self, parameters: WLParameters | Iterable[WLParameters]):
        self.iterations = 0
        self.parameters: Set[WLParameters] = {parameters} if not isinstance(parameters, Iterable) else set(parameters)
        self.init_new_parameters(self.parameters)

    @abstractmethod
    def init_new_parameters(self, parameters: Set[WLParameters]):
        pass

    @abstractmethod
    def before_update(self):
        """
        Callback before any parameters updates.
        """
        pass

    def update(self):
        """
        Main method for updating parameters.
        """
        self.before_update()
        result = self.update_body()
        self.after_update()
        return result

    @abstractmethod
    def after_update(self):
        """
        Callback after any parameters updates.
        """
        pass

    @abstractmethod
    def update_body(self):
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

    def reset_parameters(self, dump=True) -> Set[WLParameters] | None:
        params = self.parameters if dump else None
        self.parameters = set()
        return params

    def add_parameters(self, parameters: WLParameters):
        self.init_new_parameters({parameters})
        self.parameters.add(parameters)

    def update_parameters(self, parameters: WLParameters | Iterable[WLParameters]):
        parameters = parameters if isinstance(parameters, Iterable) else {parameters}
        self.init_new_parameters(parameters)
        self.parameters.update(parameters)

    def remove_parameters(self, parameters: WLParameters | Iterable[WLParameters]) -> tuple[int, Set[WLParameters]]:
        parameters = parameters if isinstance(parameters, Iterable) else {parameters}
        n_removed, not_removed = 0, set()
        for parameter in parameters:
            try:
                self.parameters.remove(parameter)
                n_removed += 1
            except KeyError:
                not_removed.add(parameter)
        return n_removed, not_removed


# SGD Optimizer with optional momentum (todo add Nesterov momentum?)
# todo add weight decay handling!
class SGD(Optimizer):

    def __init__(self, parameters: WLParameters | Iterable[WLParameters],
                 lr=0.1, lr_decay_func: Callable[[int, float], float] = None,
                 # todo maybe write a Scheduler, like in PyTorch
                 weight_decay=0., momentum=0.):
        self.lr = lr
        self.current_lr = lr
        self.lr_decay_func = lr_decay_func
        self.weight_decay = weight_decay
        self.momentum = momentum
        super(SGD, self).__init__(parameters)

    def init_new_parameters(self, parameters: Set[WLParameters]):
        if self.momentum != 0.:
            for parameter in parameters:
                parameter = cast(WLParameters, parameter)
                if not hasattr(parameter, 'weight_momentums'):
                    parameter.weight_momentums = np.zeros_like(parameter.get_weights(copy=False))
                    parameter.bias_momentums = np.zeros_like(parameter.get_biases(copy=False))

    def before_update(self):
        if self.lr_decay_func is not None:
            self.current_lr = self.lr_decay_func(self.iterations, self.current_lr)

    def update_body(self):
        weight_updates = {}
        bias_updates = {}
        if self.momentum != 0.:
            # Build updates for SGD with momentum
            for parameter in self.parameters:
                # Build weights updates
                w_updates = self.momentum * parameter.weight_momentums - \
                            self.current_lr * parameter.get_dweights(copy=False)
                weight_updates[parameter] = w_updates
                parameter.weight_momentums = w_updates

                # Build biases updates
                b_updates = self.momentum * parameter.bias_momentums - \
                            self.current_lr * parameter.get_dbiases(copy=False)
                bias_updates[parameter] = b_updates
                parameter.bias_momentums = b_updates
        else:
            # Build updates for "vanilla" SGD (i.e., without momentum)
            for parameter in self.parameters:
                w_updates = - self.current_lr * parameter.get_dweights(copy=False)
                b_updates = - self.current_lr * parameter.get_dbiases(copy=False)
                weight_updates[parameter] = w_updates
                bias_updates[parameter] = b_updates

        # Update weights and biases using either 'vanilla' updates or momentum ones
        for parameter in self.parameters:
            parameter.update_weights_and_biases(weight_updates[parameter], bias_updates[parameter])

    def after_update(self):
        self.iterations += 1


__all__ = [
    'Optimizer',
    'SGD',
]
