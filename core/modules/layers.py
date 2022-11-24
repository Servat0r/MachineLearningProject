# Base layers for a Neural Network
from __future__ import annotations
from core.utils import *
import core.functions as cf
from core.modules.regularization import *


class Layer:

    def __init__(self, frozen=False):
        """
        :param frozen: If True, sets this layer to 'frozen',
        i.e. weights/biases will NOT be updated during training.
        """
        self.input = None
        self.output = None
        self.frozen = frozen

    def freeze_layer(self):
        self.frozen = True

    def unfreeze_layer(self):
        self.frozen = False

    @abstractmethod
    def is_parametrized(self) -> bool:
        """
        Returns True iff layers contains any parameter (weights, biases) and is NOT frozen.
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

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class Sequential(Layer):
    """
    A 'container' layer (inspired by torch.nn.Sequential) that maintains a list
    of layers to which layer methods will be applied.
    """

    def __init__(self, layers: Sequence[Layer], frozen=False):
        super(Sequential, self).__init__(frozen=frozen)
        self.layers = layers

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, item):
        return self.layers[item]

    def is_parametrized(self) -> bool:
        return any(layer.is_parametrized() for layer in self.layers) and not self.frozen

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        current_output = x
        for layer in self.layers:
            current_output = layer.forward(current_output)
        self.output = current_output
        return self.output

    def backward(self, dvals: np.ndarray):
        current_dvals = dvals
        for i in range(len(self)):
            layer = self.layers[len(self) - 1 - i]
            current_dvals = layer.backward(current_dvals)
        return current_dvals


class Input(Layer):

    def __init__(self, frozen=False):
        super(Input, self).__init__(frozen=frozen)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.output = x
        return self.output

    def backward(self, dvals: np.ndarray):
        return dvals

    def is_parametrized(self) -> bool:
        return False


class Linear(Layer):
    """
    A linear layer: accepts inputs of the shape (l, 1, in_features) and returns outputs
    of the shape (l, 1, out_features).
    """
    def __init__(self, in_features: int, out_features: int, weights_initializer: Initializer,
                 biases_initializer: Initializer = ZeroInitializer(), grad_reduction='mean', frozen=False,
                 weights_regularizer: Regularizer = None, biases_regularizer: Regularizer = None, dtype=np.float64):
        """
        :param in_features: Input dimension.
        :param out_features: Output dimension.
        :param weights_initializer: Initializer to use for weights initialization.
        :param biases_initializer: Initializer to use for biases initialization. Defaults
        to a ZeroInitializer.
        :param grad_reduction: Reduction to apply to the gradients of a batch of
        examples. Can be either 'none' (no reduction, if one wants to apply a 'custom'
        one), 'sum' (sum over all the gradients), 'mean' (average over all the gradients).
        :param weights_regularizer: Regularizer to apply to layer weights.
        :param biases_regularizer: Regularizer to apply to layer biases.
        :param dtype: Numpy datatype for weights and biases. Defaults to numpy.float64.
        """
        super(Linear, self).__init__(frozen=frozen)

        # Set Layer "core" parameters
        self.in_features = in_features
        self.out_features = out_features
        weights_shape = (self.in_features, self.out_features)
        biases_shape = (1, self.out_features)
        self.weights = weights_initializer(weights_shape, dtype=dtype)
        self.biases = biases_initializer(biases_shape, dtype=dtype)
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

    def get_weights(self, copy=True) -> np.ndarray:
        return self.weights.copy() if copy else self.weights

    def get_biases(self, copy=True) -> np.ndarray | None:
        return self.biases.copy() if copy else self.biases

    def get_dweights(self, copy=True) -> np.ndarray:
        return self.dweights.copy() if copy else self.dweights

    def get_dbiases(self, copy=True) -> np.ndarray | None:
        return self.dbiases.copy() if copy else self.dbiases

    def check_input_shape(self, shape: int | Sequence) -> bool:
        if isinstance(shape, int) or len(shape) == 1:
            return True
        elif len(shape) == 2:
            return shape[1] == self.in_features
        else:
            return (shape[1] == 1) and (shape[-1] == self.in_features)

    def normalize_input_shape(self, x: np.ndarray):
        if not self.check_input_shape(x.shape):
            raise ValueError(f"Invalid input shape: expected one of l, (l,), (l, {self.in_features}), "
                             f"(l > 0, 1, {self.in_features}), got {x.shape}.")
        shape = (x.shape,) if isinstance(x.shape, int) else x.shape
        if len(shape) == 1:
            return x.reshape((shape[0], 1, 1))
        elif len(shape) == 2:
            return x.reshape((shape[0], 1, shape[1]))
        else:
            return x

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = self.normalize_input_shape(x)
        self.output = self.input @ self.weights + self.biases
        return self.output

    def backward(self, dvals: np.ndarray):
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
class Activation(Layer):
    """
    Represents an activation function layer: accepts an input of the shape (l, 1, n)
    and returns an output of the same shape after having applied an activation function.
    """
    def is_parametrized(self) -> bool:
        return False


class Sigmoid(Activation):

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.output = 1.0 / (1.0 + np.exp(-x))
        return self.output

    def backward(self, dvals: np.ndarray):
        return dvals * self.output * (1 - self.output)


class Tanh(Activation):

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.output = np.tanh(x)
        return self.output

    def backward(self, dvals: np.ndarray):
        return dvals * (1. - np.square(self.output))


class ReLU(Activation):

    def default_subgrad_func(self, dvals: np.ndarray):
        """
        Default subgradient for ReLU: for each input component,
        1 iff it is > 0, else 0; this is then multiplied by the
        dvals input to get out_dvals.
        """
        out_dvals = dvals.copy()
        out_dvals[self.input <= 0] = 0
        return out_dvals

    def __init__(self, frozen=False, subgrad_func: Callable = None):
        super(ReLU, self).__init__(frozen=frozen)
        self.subgrad_func = self.default_subgrad_func if subgrad_func is None else subgrad_func

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.output = np.maximum(self.input, 0)
        return self.output

    def backward(self, dvals: np.ndarray):
        return self.subgrad_func(dvals)


class SoftmaxLayer(Activation):

    def __init__(self, const_shift=0, max_shift=False):
        super(SoftmaxLayer, self).__init__()
        self.softmax = cf.Softmax(const_shift=const_shift, max_shift=max_shift)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.output = self.softmax(x)
        return self.output

    # todo check!
    def backward(self, dvals: np.ndarray):
        return self.softmax.vjp(self.input, dvals)


class Dense(Layer):
    """
    A fully-connected layer with activation function for all the neurons.
    """
    def __init__(
            self, in_features: int, out_features: int, activation_layer: Activation,
            weights_initializer: Initializer, biases_initializer: Initializer = ZeroInitializer(),
            grad_reduction='mean', frozen=False, weights_regularizer: Regularizer = None,
            biases_regularizer: Regularizer = None, dtype=np.float64
    ):
        super(Dense, self).__init__(frozen=frozen)
        # Initialize linear part
        self.linear = Linear(
            in_features, out_features, weights_initializer, biases_initializer,
            grad_reduction, weights_regularizer=weights_regularizer,
            biases_regularizer=biases_regularizer, dtype=dtype,
        )
        self.activation = activation_layer
        self.net = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.net = self.linear.forward(x)
        self.output = self.activation.forward(self.net)
        return self.output

    def backward(self, dvals: np.ndarray):
        dvals = self.activation.backward(dvals)
        return self.linear.backward(dvals)

    def is_parametrized(self) -> bool:
        return self.linear.is_parametrized() and not self.frozen


__all__ = [
    'Layer',
    'Input',
    'Sequential',
    'Linear',
    'Activation',
    'Dense',
    'Sigmoid',
    'Tanh',
    'ReLU',
    'SoftmaxLayer',
]
