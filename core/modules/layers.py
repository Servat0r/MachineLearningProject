# Base layers for a Neural Network
from __future__ import annotations
from core.utils import *
import core.diffs as dfs
import core.functions as cf
from .parameters import *
from .regularization import *


class Layer:

    def __init__(self):
        self.__is_training = False
        self.input = None
        self.output = None

    @abstractmethod
    def check_input_shape(self, shape: int | Sequence) -> bool:
        """
        Checks shape of the input to the layer; must be of the form (l, <data_shape>)
        with l = #inputs given.
        :param shape: Shape to check.
        :return: True if shape has valid form, False otherwise.
        """
        pass

    @abstractmethod
    def check_backward_input_shape(self, shape: int | Sequence) -> bool:
        """
        Checks shape of the "backward inputs" of the layer. Generally, they shall be of the shape
        (l, n, 1), with l = batch size, n = backward input size.
        :param shape: Shape to check.
        :return: True if shape has valid form, False otherwise.
        """
        pass

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the current level.
        :param x:
        :return:
        """
        pass

    @abstractmethod
    def backward(self, dvals: np.ndarray):
        pass

    def clear(self):
        self.input = None
        self.output = None

    @abstractmethod
    def get_parameters(self) -> Set[Parameters]:
        pass

    def set_to_train(self):
        self.__is_training = True

    def set_to_eval(self):
        self.__is_training = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class WeightedLayer(Layer):

    """
    Base class for layers that include weights and biases.
    """
    def __init__(self, initializer: Initializer, init_args: dict[str, Any] = None,
                 regularizers: Regularizer | Iterable[Regularizer] = None):
        super(WeightedLayer, self).__init__()
        self.weights, self.biases = self._initialize_weights(initializer, init_args=init_args)
        self.parameters = WeightedLayerParameters(
            self.weights, self.biases, grad_reduction=WeightedLayerParameters.SUM,
        )
        self.regularizers = None
        if regularizers is not None:
            regularizers = {regularizers} if not isinstance(regularizers, Iterable) else regularizers
            self.regularizers = set(regularizers)
            for regularizer in self.regularizers:
                regularizer.init_new_parameters(self.get_parameters())

    def get_weights(self, copy=True) -> np.ndarray:
        """
        Gets the weights of the current level as a matrix, with rows corresponding
        to neurons.
        :return: A (multi-dimensional) array containing the weights of the layer.
        """
        return self.weights.copy() if copy else self.weights

    def get_biases(self, copy=True) -> np.ndarray | None:
        """
        Gets the biases of the current level as an array, if existing.
        :return: An array containing the biases of the layer, if existing.
        """
        return self.biases.copy() if copy else self.biases

    def get_dweights(self, copy=True) -> np.ndarray:
        return self.parameters.get_dweights(copy)

    def get_dbiases(self, copy=True) -> np.ndarray | None:
        return self.parameters.get_dbiases(copy)

    @abstractmethod
    def _initialize_weights(self, initializer: Initializer,
                            init_args: dict[str, Any] = None) -> tuple[np.ndarray, Optional[np.ndarray]]:
        pass

    def get_parameters(self) -> Set[Parameters]:
        return {self.parameters}


class SequentialLayer(Layer):

    def __init__(self, layers: Sequence[Layer]):
        super(SequentialLayer, self).__init__()
        self.layers = layers

    def __len__(self):
        return len(self.layers)

    def check_input_shape(self, shape: int | Sequence) -> bool:
        if isinstance(shape, int):
            return False
        return (len(shape) > 0) and all([sdim > 0 for sdim in shape])

    def check_backward_input_shape(self, shape: int | Sequence) -> bool:
        return self.layers[-1].check_backward_input_shape(shape)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        current_output = x
        for layer in self.layers:
            current_output = layer.forward(current_output)
        self.output = current_output
        return self.output

    def backward(self, dvals: np.ndarray):
        if not self.check_backward_input_shape(dvals.shape):
            raise ValueError(f"Invalid shape {dvals.shape}")    # we don't know the shape accepted by the last layer
        current_dvals = dvals
        for i in range(len(self)):
            layer = self.layers[len(self) - 1 - i]
            current_dvals = layer.backward(current_dvals)
        return current_dvals

    def get_parameters(self) -> Set[Parameters]:
        parameters = set()
        for layer in self.layers:
            parameters.update(layer.get_parameters())
        return parameters


class LinearLayer(WeightedLayer):
    """
    A linear layer: accepts inputs of the shape (l, 1, in_features) and returns outputs
    of the shape (l, 1, out_features).
    """

    def __init__(self, initializer: Initializer, in_features: int,
                 out_features: int, init_args: dict[str, Any] = None,
                 regularizers: Regularizer | Iterable[Regularizer] = None):
        self.in_features = in_features
        self.out_features = out_features
        super(LinearLayer, self).__init__(initializer, init_args=init_args, regularizers=regularizers)

    def _initialize_weights(self, initializer: Initializer, init_args: dict[str, Any] = None):
        weights_shape = (self.in_features, self.out_features)
        biases_shape = (1, self.out_features)
        init_args = init_args if init_args is not None else {}
        return initializer(weights_shape, biases_shape, **init_args)

    def check_input_shape(self, shape: int | Sequence) -> bool:
        if isinstance(shape, int):
            return False
        return all([len(shape) == 3, shape[0] > 0, shape[1] == 1, shape[2] == self.in_features])

    def check_backward_input_shape(self, shape: int | Sequence) -> bool:
        if isinstance(shape, int):
            return False
        return all([len(shape) == 3, shape[0] > 0, shape[1] == 1, shape[2] == self.out_features])
        # todo now the above is only for row vectors

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.check_input_shape(x.shape):
            raise ValueError(f"Invalid input shape: expected (l > 0, 1, {self.in_features}), got {x.shape}.")
        self.input = x
        self.output = self.input @ self.weights + (self.biases if self.biases is not None else 0)
        return self.output

    def backward(self, dvals: np.ndarray):
        if not self.check_backward_input_shape(dvals.shape):
            raise ValueError(f"Invalid input shape: expected (l > 0, 1, {self.out_features}), got {dvals.shape}")
        # Calculate update to layer's weights and biases
        dvals2 = np.transpose(dvals, axes=[0, 2, 1])    # (l, n, 1)
        tinp = np.transpose(self.input, axes=[0, 2, 1])  # (l, m, 1)
        dbiases = dvals   # (l, 1, n) todo need to make a copy?
        dweights = tinp @ dvals  # (l, m, 1) * (l, 1, n)
        self.parameters.update_grads(dweights, dbiases)
        # Now calculate values to backpropagate to previous layer
        out_dvalues = np.transpose(self.weights @ dvals2, axes=[0, 2, 1])
        # Handle regularization
        # todo need to modify because like this, if a regularizer is global then param
        # todo grads are computed multiple times!
        if self.regularizers is not None:
            for regularizer in self.regularizers:
                regularizer.update_param_grads(layer=self)
        return out_dvalues


class ActivationLayer(Layer):
    """
    Represents an activation function layer: accepts an input of the shape (l, 1, n)
    and returns an output of the same shape after having applied an activation function.
    """
    def __init__(self, func: Callable):
        super(ActivationLayer, self).__init__()
        self.func = func
        self.func_is_type = isinstance(func, type)  # for classes

    def check_input_shape(self, shape: int | Sequence) -> bool:
        if isinstance(shape, int):
            return False
        return all([len(shape) == 3, shape[0] > 0, shape[1] == 1, shape[2] > 0])

    def check_backward_input_shape(self, shape: int | Sequence) -> bool:
        if isinstance(shape, int):
            return False
        return all([len(shape) == 3, shape[0] > 0, shape[1] == 1, shape[2] > 0])

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.check_input_shape(x.shape):
            raise ValueError(f"Invalid shape: expected (l > 0, 1, n > 0), got {x.shape}")
        self.input = x
        self.output = self.func(self.input)
        return self.output

    def backward(self, dvals: np.ndarray):
        if not self.check_backward_input_shape(dvals.shape):
            raise ValueError(f"Invalid shape: expected (l > 0, 1, n > 0), got {dvals.shape}")
        if self.func_is_type:
            return dfs.vjp(type(self.func), self.func, self.input, dvals)
        else:
            return dfs.vjp(self.func, self.input, dvals)

    def get_parameters(self) -> Set[Parameters]:
        return set()    # we don't have parameters here!


class FullyConnectedLayer(Layer):
    """
    A fully-connected layer with activation function for all the neurons.
    """
    def __init__(
            self, in_features: int, out_features: int, activation_layer: ActivationLayer = None,
            func: Callable = None, initializer: Initializer = None, init_args: dict[str, Any] = None,
            regularizers: Regularizer | Iterable[Regularizer] = None,
    ):
        super(FullyConnectedLayer, self).__init__()
        # Initialize linear part
        self.linear = LinearLayer(initializer, in_features, out_features, init_args, regularizers)
        self.activation = None
        if activation_layer is not None:
            self.activation = activation_layer
        else:
            self.activation = ActivationLayer(func)

    def check_input_shape(self, shape: int | Sequence) -> bool:
        return self.linear.check_input_shape(shape)

    def check_backward_input_shape(self, shape: int | Sequence) -> bool:
        return self.activation.check_backward_input_shape(shape)

    def forward(self, x: np.ndarray) -> np.ndarray:
        net = self.linear.forward(x)
        self.output = self.activation.forward(net)
        return self.output

    def backward(self, dvals: np.ndarray):
        dvals = self.activation.backward(dvals)  # net is actually saved as input to the activation layer
        return self.linear.backward(dvals)

    def get_parameters(self) -> Set[Parameters]:
        return self.linear.get_parameters().union(self.activation.get_parameters())


class SignLayer(ActivationLayer):

    def __init__(self):
        super(SignLayer, self).__init__(func=cf.sign)


class SigmoidLayer(ActivationLayer):

    def __init__(self):
        super(SigmoidLayer, self).__init__(func=cf.sigmoid)


class TanhLayer(ActivationLayer):

    def __init__(self):
        super(TanhLayer, self).__init__(func=cf.tanh)


class ReLULayer(ActivationLayer):

    def __init__(self):
        super(ReLULayer, self).__init__(func=cf.relu)


class SoftmaxLayer(ActivationLayer):

    def __init__(self, const_shift=0, max_shift=False):
        func = cf.Softmax(const_shift, max_shift)
        super(SoftmaxLayer, self).__init__(type(func))
        self.func = func


__all__ = [
    'Layer',
    'WeightedLayer',
    'SequentialLayer',
    'LinearLayer',
    'ActivationLayer',
    'FullyConnectedLayer',
    'SignLayer',
    'SigmoidLayer',
    'TanhLayer',
    'ReLULayer',
    'SoftmaxLayer',
]
