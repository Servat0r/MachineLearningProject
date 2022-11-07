# Base layers for a Neural Network
from __future__ import annotations
from core.utils.types import *
from core.utils import Initializer, ZeroInitializer
from core.functions import *


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
    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Forward pass of the current level.
        :param input:
        :return:
        """
        pass

    @abstractmethod
    def backward(self, dvals: np.ndarray):
        pass

    def clear(self):
        self.input = None
        self.output = None

    def set_to_train(self):
        self.__is_training = True

    def set_to_eval(self):
        self.__is_training = False

    def __call__(self, input: np.ndarray) -> np.ndarray:
        return self.forward(input)


class WeightedLayer(Layer):

    """
    Base class for layers that include weights and biases.
    """
    def __init__(self, initializer: Initializer, init_args: dict[str, Any] = None):
        super(WeightedLayer, self).__init__()
        self.weights, self.biases = self._initialize_weights(initializer, init_args=init_args)
        # Gradients for weights updating
        self.dweights, self.dbiases = self._initialize_weights(ZeroInitializer(), init_args=init_args)

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
        return self.dweights.copy() if copy else self.dweights

    def get_dbiases(self, copy=True) -> np.ndarray | None:
        return self.dbiases.copy() if copy else self.dbiases

    @abstractmethod
    def _initialize_weights(self, initializer: Initializer,
                            init_args: dict[str, Any] = None) -> tuple[np.ndarray, Optional[np.ndarray]]:
        pass


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

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        current_output = input
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


class LinearLayer(WeightedLayer):
    """
    A linear layer: accepts inputs of the shape (l, 1, in_features) and returns outputs
    of the shape (l, 1, out_features).
    """

    def __init__(self, initializer: Initializer, in_features: int, out_features: int, init_args: dict[str, Any] = None):
        self.in_features = in_features
        self.out_features = out_features
        super(LinearLayer, self).__init__(initializer, init_args=init_args)

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
        return all([len(shape) == 3, shape[0] > 0, shape[1] == self.out_features, shape[2] == 1])

    def forward(self, input: np.ndarray) -> np.ndarray:
        if not self.check_input_shape(input.shape):
            raise ValueError(f"Invalid input shape: expected (l > 0, 1, {self.in_features}), got {input.shape}.")
        self.input = input
        self.output = self.input @ self.weights + (self.biases if self.biases is not None else 0)
        return self.output

    def backward(self, dvals: np.ndarray):
        if not self.check_backward_input_shape(dvals.shape):
            raise ValueError(f"Invalid input shape: expected (l > 0, {self.out_features}, 1), got {dvals.shape}")
        # Calculate update to layer's weights and biases
        dshape = dvals.shape
        self.dweights = dvals @ self.input
        self.dbiases = np.reshape(dvals, (dshape[0], dshape[2], dshape[1]))
        # Now calculate values to backpropagate to previous layer
        out_dvalues = self.weights @ dvals
        return out_dvalues


class ActivationLayer(Layer):
    """
    Represents an activation function layer: accepts an input of the shape (l, 1, n)
    and returns an output of the same shape after having applied an activation function.
    """
    def __init__(self, func: ActivationFunction):
        super(ActivationLayer, self).__init__()
        self.func = func

    def check_input_shape(self, shape: int | Sequence) -> bool:
        if isinstance(shape, int):
            return False
        return all([len(shape) == 3, shape[0] > 0, shape[1] == 1, shape[2] > 0])

    def check_backward_input_shape(self, shape: int | Sequence) -> bool:
        if isinstance(shape, int):
            return False
        return all([len(shape) == 3, shape[0] > 0, shape[1] > 0, shape[2] == 1])

    def forward(self, input: np.ndarray) -> np.ndarray:
        if not self.check_input_shape(input.shape):
            raise ValueError(f"Invalid shape: expected (l > 0, 1, n > 0), got {input.shape}")
        self.input = input
        self.output = self.func(self.input)
        return self.output

    def backward(self, dvals: np.ndarray):
        if not self.check_backward_input_shape(dvals.shape):
            raise ValueError(f"Invalid shape: expected (l > 0, n > 0, 1), got {dvals.shape}")
        inshape = self.input.shape
        y = np.reshape(self.func.grad()(self.input), (inshape[0], inshape[2], inshape[1]))
        return dvals * y


class FullyConnectedLayer(LinearLayer):
    """
    A fully-connected layer with activation function for all the neurons.
    """
    def __init__(self, in_features: int, out_features: int, func: ActivationFunction,
                 initializer: Initializer, init_args: dict[str, Any] = None):
        # Initialize linear part
        super(FullyConnectedLayer, self).__init__(initializer, in_features, out_features, init_args)
        # Activation part
        self.func = func
        self.net = None

    def check_input_shape(self, shape: int | Sequence) -> bool:
        return super(FullyConnectedLayer, self).check_input_shape(shape)

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.net = super(FullyConnectedLayer, self).forward(input)
        self.output = self.func(self.net)
        return self.output

    def backward(self, dvals: np.ndarray):
        inshape = self.net.shape
        y = np.reshape(self.func.grad()(self.net), (inshape[0], inshape[2], inshape[1]))
        dvals = dvals * y
        return super(FullyConnectedLayer, self).backward(dvals)


class SignLayer(ActivationLayer):

    def __init__(self):
        super(SignLayer, self).__init__(func=Sign())


class SigmoidLayer(ActivationLayer):

    def __init__(self):
        super(SigmoidLayer, self).__init__(func=Sigmoid())


class TanhLayer(ActivationLayer):

    def __init__(self):
        super(TanhLayer, self).__init__(func=Tanh())


class ReLULayer(ActivationLayer):

    def __init__(self):
        super(ReLULayer, self).__init__(func=ReLU())


class SoftmaxLayer(Layer):

    def check_input_shape(self, shape: int | Sequence) -> bool:
        if isinstance(shape, int):
            return False
        return all([len(shape) == 3, shape[0] > 0, shape[1] == 1, shape[2] > 0])

    def check_backward_input_shape(self, shape: int | Sequence) -> bool:
        if isinstance(shape, int):
            return False
        return all([len(shape) == 3, shape[0] > 0, shape[1] > 0, shape[2] == 1])

    def forward(self, input: np.ndarray) -> np.ndarray:
        if not self.check_input_shape(input.shape):
            raise ValueError(f"Invalid shape: expected (l > 0, 1, n > 0), got {input.shape}")
        self.input = input
        self.output = Softmax()(self.input)
        return self.output

    def backward(self, dvals: np.ndarray):
        if not self.check_backward_input_shape(dvals.shape):
            raise ValueError(f"Invalid shape: expected (l > 0, n > 0, 1), got {dvals.shape}")
        y = Softmax().grad()(self.input)  # (l, n, n)
        return y @ dvals    # (l, n, 1)


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