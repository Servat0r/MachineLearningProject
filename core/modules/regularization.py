# Regularized losses (L1, L2 etc.)
from __future__ import annotations
from ..utils import *
from .parameters import Parameters, WeightedLayerParameters as WLParameters


class Regularizer(Callable):
    """
    Base regularizator, applicable to a set of parameters.
    """
    def __init__(self, parameters: Parameters | Iterable[Parameters] = None):
        if parameters is not None:
            parameters = {parameters} if not isinstance(parameters, Iterable) else set(parameters)
        self.parameters = parameters if parameters is not None else set()
        if self.parameters is not None:
            self.init_new_parameters(self.parameters)

    @property
    def key(self) -> str:
        return type(self).__name__

    def init_new_parameters(self, parameters: Set[Parameters]):
        for parameter in parameters:
            if not hasattr(parameter, 'regularizer_updates'):
                parameter.regularizer_updates = {self.key: {'weights': None, 'biases': None}}
            elif parameter.regularizer_updates.get(self.key) is not None:
                raise RuntimeError(f"Attempted to add more than one different '{self.key}' regularizers!")
            else:
                parameter.regularizer_updates[self.key] = {'weights': None, 'biases': None}
            parameter.regularizers.add(self)
            self.parameters.add(parameter)

    @abstractmethod
    def __call__(self, target_shape: tuple = (1, 1), result_arr: np.ndarray = None) -> np.ndarray:
        pass

    def add_parameters(self, parameters: WLParameters | Iterable[WLParameters]):
        parameters = {parameters} if not isinstance(parameters, Iterable) else parameters
        self.init_new_parameters(parameters)
        self.parameters.update(parameters)

    def remove_parameters(self, parameters: WLParameters | Iterable[WLParameters]):
        parameters = {parameters} if not isinstance(parameters, Iterable) else parameters
        n_removed, not_removed = 0, set()
        for parameter in parameters:
            try:
                self.parameters.remove(parameter)
                parameter.regularizers.remove(self)
                parameter.regularized_updates.pop(self.key, None)
                n_removed += 1
            except KeyError:
                not_removed.add(parameter)
        return n_removed, not_removed

    def reset_parameters(self):
        for parameter in self.parameters:
            parameter.regularized_updates.pop(self.key, None)
            parameter.regularizers.remove(self)
        self.parameters = set()

    @abstractmethod
    def update_param_grads(self, layer=None):
        pass


class L1Regularizer(Regularizer):
    """
    L1 Regularizator, with the possibility to vary the subgradient used.
    """
    @staticmethod
    def default_subgrad_func(parameter: WLParameters):
        """
        Default subgradient function for L1 regularization. Given a set of Parameters
        with their weights, it returns the absolute value of the sign of each weight.
        """
        weights, biases = parameter.get_weights(), parameter.get_biases()  # Here we copy weights and biases
        weights = np.sign(weights)
        biases = np.sign(biases)
        return weights, biases

    def __init__(
            self, parameters: WLParameters | Iterable[WLParameters] = None,
            subgrad_func: Callable = None, l1_lambda: float = 0.1,
    ):
        super(L1Regularizer, self).__init__(parameters)
        self.subgrad_func = subgrad_func if subgrad_func is not None else self.default_subgrad_func
        self.l1_lambda = l1_lambda

    def update_param_grads(self, layer=None):
        parameters = self.parameters if layer is None else list(filter(lambda par: par.layer == layer, self.parameters))
        for parameter in parameters:
            weights, biases = self.subgrad_func(parameter)
            parameter.regularized_updates[self.key]['weights'] = self.l1_lambda * weights
            parameter.regularized_updates[self.key]['biases'] = self.l1_lambda * biases

    def __call__(self, target_shape: tuple = (1, 1), result_arr: np.ndarray = None):
        result = np.zeros(target_shape) if result_arr is None else result_arr
        val = 0.
        for parameter in self.parameters:
            val += np.sum(np.abs(parameter.get_weights(copy=False)))
            val += np.sum(np.abs(parameter.get_biases(copy=False)))  # todo do we sum also biases?
        val *= self.l1_lambda
        result += val
        return result


class L2Regularizer(Regularizer):
    """
    L2 regularizator.
    """
    def __init__(
            self, parameters: WLParameters | Iterable[WLParameters] = None, l2_lambda: float = 0.1,
    ):
        super(L2Regularizer, self).__init__(parameters)
        self.l2_lambda = l2_lambda

    def update_param_grads(self, layer=None):
        parameters = self.parameters if layer is None else list(filter(lambda par: par.layer == layer, self.parameters))
        for parameter in parameters:
            weights = 2 * self.l2_lambda * parameter.get_weights(copy=False)
            biases = 2 * self.l2_lambda * parameter.get_biases(copy=False)
            parameter.regularized_updates[self.key]['weights'] = weights
            parameter.regularized_updates[self.key]['biases'] = biases

    def __call__(self, target_shape: tuple = (1, 1), result_arr: np.ndarray = None) -> np.ndarray:
        result = np.zeros(target_shape) if result_arr is None else result_arr
        val = 0.
        for parameter in self.parameters:
            val += np.square(np.linalg.norm(parameter.get_weights(copy=False)))
            val += np.square(np.linalg.norm(parameter.get_biases(copy=False)))  # todo do we sum also biases?
        val *= self.l2_lambda
        result += val
        return result


__all__ = [
    'Regularizer',
    'L1Regularizer',
    'L2Regularizer',
]
