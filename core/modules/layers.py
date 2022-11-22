# Base layers for a Neural Network
from __future__ import annotations
from core.utils import *
from core.modules.regularization import *


class Layer:

    def __init__(self, frozen=False):
        self.__is_training = False
        self.input = None
        self.output = None
        self.frozen = frozen

    def freeze_layer(self):
        self.frozen = True

    def unfreeze_layer(self):
        self.frozen = False

    @abstractmethod
    def is_parametrized(self) -> bool:
        pass

    @abstractmethod
    def update_reg_values(self, func: Callable[np.ndarray]):
        pass

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

    def set_to_train(self):
        self.__is_training = True

    def set_to_eval(self):
        self.__is_training = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class SequentialLayer(Layer):

    def __init__(self, layers: Sequence[Layer], frozen=False):
        super(SequentialLayer, self).__init__(frozen=frozen)
        self.layers = layers

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, item):
        return self.layers[item]

    def is_parametrized(self) -> bool:
        return any(layer.is_parametrized() for layer in self.layers) and not self.frozen

    def update_reg_values(self, func: Callable[np.ndarray]):
        for layer in self.layers:
            layer.update_reg_values(func)

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
        # Handle regularizers from underlying layers
        # todo In this implementation there is no control if the regularizer are also in
        # todo OTHER models; a safer (yet more costly) way should be that of maintaining
        # todo an index of layers for each parameter and filter by them
        return current_dvals


class LinearLayer(Layer):
    """
    A linear layer: accepts inputs of the shape (l, 1, in_features) and returns outputs
    of the shape (l, 1, out_features).
    """

    def __init__(self, in_features: int, out_features: int,
                 initializer: Initializer, init_args: dict[str, Any] = None,
                 grad_reduction='mean', frozen=False,
                 weights_regularizer: Regularizer = None,
                 biases_regularizer: Regularizer = None,
                 ):
        super(LinearLayer, self).__init__(frozen=frozen)

        # Set Layer "core" parameters
        self.in_features = in_features
        self.out_features = out_features
        init_args = init_args if init_args is not None else {}
        weights_shape = (self.in_features, self.out_features)
        biases_shape = (1, self.out_features)
        self.weights, self.biases = initializer(weights_shape, biases_shape, **init_args)
        self.grad_reduction = grad_reduction

        # Set updates for backward
        self.dweights = None
        self.dbiases = None

        # Set updates for momentums
        self.weight_momentums = np.zeros_like(self.weights)
        self.bias_momentums = np.zeros_like(self.biases)

        # Set regularizer and its updates
        self.weights_regularizer = weights_regularizer
        self.biases_regularizer = biases_regularizer
        self.weights_reg_updates = None
        self.biases_reg_updates = None

    def is_parametrized(self) -> bool:
        return not self.frozen

    def update_reg_values(self, func: Callable[np.ndarray]):
        if self.weights_regularizer is not None:
            w_updates = func(self.weights_reg_updates)
            self.weights += w_updates
        if self.biases_regularizer is not None:
            b_updates = func(self.biases_reg_updates)
            self.biases += b_updates

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
        self.output = self.input @ self.weights + self.biases
        return self.output

    def backward(self, dvals: np.ndarray):
        if not self.check_backward_input_shape(dvals.shape):
            raise ValueError(f"Invalid input shape: expected (l > 0, 1, {self.out_features}), got {dvals.shape}")
        tinp = np.transpose(self.input, axes=[0, 2, 1])  # (l, m, 1)
        # Calculate update to layer's weights and biases
        self.dweights = tinp @ dvals
        self.dbiases = dvals.copy()

        # Apply reductions if requested
        if self.grad_reduction == 'sum':
            self.dweights = np.sum(self.dweights, axis=0)
            self.dbiases = np.sum(self.dbiases, axis=0)
        elif self.grad_reduction == 'mean':
            self.dweights = np.mean(self.dweights, axis=0)
            self.dbiases = np.mean(self.dbiases, axis=0)

        # Handle regularization
        if self.weights_regularizer is not None:
            self.weights_reg_updates = self.weights_regularizer.update(self.weights)

        if self.biases_regularizer is not None:
            self.biases_reg_updates = self.biases_regularizer.update(self.biases)

        # Now calculate values to backpropagate to previous layer
        return np.dot(dvals, self.weights.T)


# noinspection PyAbstractClass
class ActivationLayer(Layer):
    """
    Represents an activation function layer: accepts an input of the shape (l, 1, n)
    and returns an output of the same shape after having applied an activation function.
    """
    def is_parametrized(self) -> bool:
        return False

    def update_reg_values(self, func: Callable[np.ndarray]):
        pass

    def check_input_shape(self, shape: int | Sequence) -> bool:
        if isinstance(shape, int):
            return False
        return all([len(shape) == 3, shape[0] > 0, shape[1] == 1, shape[2] > 0])

    def check_backward_input_shape(self, shape: int | Sequence) -> bool:
        if isinstance(shape, int):
            return False
        return all([len(shape) == 3, shape[0] > 0, shape[1] == 1, shape[2] > 0])


class SigmoidLayer(ActivationLayer):

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.output = 1.0 / (1.0 + np.exp(-x))
        return self.output

    def backward(self, dvals: np.ndarray):
        return dvals * self.output * (1 - self.output)


class TanhLayer(ActivationLayer):

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.output = np.tanh(x)
        return self.output

    def backward(self, dvals: np.ndarray):
        return dvals * (1. - np.square(self.output))


class ReLULayer(ActivationLayer):

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.output = np.maximum(self.input, 0)
        return self.output

    def backward(self, dvals: np.ndarray):
        out_dvals = dvals.copy()
        out_dvals[self.input <= 0] = 0
        return out_dvals


class SoftmaxLayer(ActivationLayer):

    def __init__(self, const_shift=0, max_shift=False):
        super(SoftmaxLayer, self).__init__()
        self.const_shift = const_shift  # constant shift for arguments
        self.max_shift = max_shift      # subtracting the maximum input value from all inputs

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        if self.const_shift != 0:
            x += self.const_shift
        if self.max_shift:
            # Get unnormalized probabilities
            exp_values = np.exp(x - np.max(x, axis=-1, keepdims=True))
        else:
            exp_values = np.exp(x)
        s = np.sum(exp_values, axis=-1, keepdims=True)
        self.output = exp_values / s
        return self.output

    # todo check!
    def backward(self, dvals: np.ndarray):
        sd = self.output * dvals
        sd_sum = np.sum(sd, axis=1, keepdims=True)
        return sd - sd_sum * self.output


class FullyConnectedLayer(Layer):
    """
    A fully-connected layer with activation function for all the neurons.
    """
    def __init__(
            self, in_features: int, out_features: int, activation_layer: ActivationLayer,
            initializer: Initializer = None, init_args: dict[str, Any] = None,
            grad_reduction='mean', frozen=False,
            weights_regularizer: Regularizer = None,
            biases_regularizer: Regularizer = None,
    ):
        super(FullyConnectedLayer, self).__init__(frozen=frozen)
        # Initialize linear part
        self.linear = LinearLayer(
            in_features, out_features, initializer, init_args, grad_reduction,
            weights_regularizer=weights_regularizer, biases_regularizer=biases_regularizer
        )
        self.activation = activation_layer
        self.net = None

    def update_reg_values(self, func: Callable[np.ndarray]):
        self.linear.update_reg_values(func)

    def check_input_shape(self, shape: int | Sequence) -> bool:
        return self.linear.check_input_shape(shape)

    def check_backward_input_shape(self, shape: int | Sequence) -> bool:
        return self.activation.check_backward_input_shape(shape)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.net = self.linear.forward(x)
        self.output = self.activation.forward(self.net)
        return self.output

    def backward(self, dvals: np.ndarray):
        dvals = self.activation.backward(dvals)  # net is actually saved as input to the activation layer
        return self.linear.backward(dvals)

    def is_parametrized(self) -> bool:
        return self.linear.is_parametrized() and not self.frozen


__all__ = [
    'Layer',
    'SequentialLayer',
    'LinearLayer',
    'ActivationLayer',
    'FullyConnectedLayer',
    'SigmoidLayer',
    'TanhLayer',
    'ReLULayer',
    'SoftmaxLayer',
]
