# Optimizers
from __future__ import annotations
from ..utils import *
from .parameters import WeightedLayerParameters as WLParameters
from .schedulers import *


# Base class
class Optimizer:

    def __init__(self, parameters: WLParameters | Iterable[WLParameters] = None):
        self.iterations = 0
        self.parameters = set()
        if parameters is not None:
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
    def update_reg_values(self, parameter, w_vals: np.ndarray, b_vals: np.ndarray):
        """
        Subroutine for handling regularization terms in updating.
        """

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

    def remove_all_parameters(self, dump=True) -> Set[WLParameters] | None:
        params = self.parameters if dump else None
        self.parameters = set()
        return params

    def add_parameters(self, parameters: WLParameters):
        self.init_new_parameters({parameters})
        self.parameters.add(parameters)

    def update_parameters(self, parameters: WLParameters | Iterable[WLParameters]):
        parameters = {parameters} if not isinstance(parameters, Iterable) else parameters
        self.init_new_parameters(parameters)
        self.parameters.update(parameters)

    def remove_parameters(self, parameters: WLParameters | Iterable[WLParameters]) -> tuple[int, Set[WLParameters]]:
        parameters = {parameters} if not isinstance(parameters, Iterable) else parameters
        n_removed, not_removed = 0, set()
        for parameter in parameters:
            try:
                self.parameters.remove(parameter)
                n_removed += 1
            except KeyError:
                not_removed.add(parameter)
        return n_removed, not_removed

    def zero_grads(self):
        for parameter in self.parameters:
            parameter.zero_grads()


# SGD Optimizer with optional momentum (todo add Nesterov momentum?)
class SGD(Optimizer):

    def __init__(self, parameters: WLParameters | Iterable[WLParameters] = None,
                 lr=0.1, lr_decay_scheduler: Scheduler = None,
                 weight_decay=0., momentum=0.):
        self.lr = lr
        self.current_lr = lr
        self.lr_decay_scheduler = lr_decay_scheduler
        self.weight_decay = weight_decay
        self.momentum = momentum
        super(SGD, self).__init__(parameters)

    def init_new_parameters(self, parameters: Set[WLParameters]):
        if self.momentum != 0.:
            for parameter in parameters:
                if parameter.weight_momentums is None:
                    parameter.weight_momentums = np.zeros_like(parameter.get_weights(copy=False))
                    parameter.bias_momentums = np.zeros_like(parameter.get_biases(copy=False))

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
                w_vals -= regw_updates  # self.apply_reduction(regw_updates) todo sure? we are ignoring batch size!
            if regb_updates is not None:
                b_vals -= regb_updates  # self.apply_reduction(regb_updates) todo same as above
        return w_vals, b_vals   # todo necessary?

    def update_body(self):
        if self.momentum != 0.:
            # Build updates for SGD with momentum
            for parameter in self.parameters:
                # Build weights updates
                parameter.weight_momentums[:] = self.momentum * parameter.weight_momentums - \
                            self.current_lr * parameter.get_dweights(copy=False)
                # Build biases updates
                parameter.bias_momentums[:] = self.momentum * parameter.bias_momentums - \
                            self.current_lr * parameter.get_dbiases(copy=False)

                # Handle regularizations
                # w_updates, b_updates = self.update_reg_values(parameter, w_updates, b_updates)
                self.update_reg_values(parameter, parameter.weight_momentums[:], parameter.bias_momentums[:])

                # Handle weight decay case (i.e., implicit L2 regul.) todo eliminate!
                if self.weight_decay != 0.:
                    parameter.weight_momentums += self.weight_decay * parameter.get_weights(copy=False)
                    parameter.bias_momentums += self.weight_decay * parameter.get_biases(copy=False)

                # Update weights and biases using momentum updates
                parameter.update_weights_and_biases(parameter.weight_momentums, parameter.bias_momentums)
        else:
            # Build updates for "vanilla" SGD (i.e., without momentum)
            for parameter in self.parameters:
                w_updates = - self.current_lr * parameter.get_dweights(copy=False)
                b_updates = - self.current_lr * parameter.get_dbiases(copy=False)

                # Handle regularizations
                w_updates, b_updates = self.update_reg_values(parameter, w_updates, b_updates)

                parameter.update_weights_and_biases(w_updates, b_updates)

    def after_update(self):
        self.iterations += 1


__all__ = [
    'Optimizer',
    'SGD',
]
