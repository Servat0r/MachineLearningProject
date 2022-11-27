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
        self.frozen = frozen
        self.__is_training = None
        self.input = None
        self.output = None

    def _init_unpickled_params(self):
        self.input = None
        self.output = None
        self.__is_training = None

    def freeze_layer(self):
        self.frozen = True

    def unfreeze_layer(self):
        self.frozen = False

    def set_to_train(self):
        self.__is_training = True

    def set_to_eval(self):
        self.__is_training = False

    def is_training(self):
        return self.is_trainable() and self.__is_training

    @abstractmethod
    def is_trainable(self) -> bool:
        """
        Returns True iff layers contains any parameter (weights, biases) and is NOT frozen.
        """
        pass

    def get_parameters(self) -> dict:
        """
        Retrieves layer parameters as a dictionary ('weights', 'biases', 'dweights' etc.).
        """
        return {}   # By default, no parameters for this layer

    def set_parameters(self, param_dict: dict):
        if not isinstance(param_dict, dict):
            raise TypeError(f"Wrong type {type(param_dict)} when setting parameters for {self}")
        for param_name, param_val in param_dict.items():
            if isinstance(param_val, np.ndarray):
                setattr(self, param_name, param_val)
            elif isinstance(param_val, dict):
                attr = getattr(self, param_name)
                if attr is None:
                    raise AttributeError(f"Unknown attribute {param_name} for {type(self)}")
                elif not isinstance(attr, Layer):
                    raise TypeError(f"{attr} is not an instance of {Layer}")
                else:
                    attr.set_parameters(param_val)
            else:
                raise TypeError(f"Wrong type {type(param_val)} for {param_name} when parsing param dict")

    def __getstate__(self):
        return {
            'frozen': self.frozen,
        }

    def __setstate__(self, state):
        frozen = state.pop('frozen', False)
        self.frozen = frozen
        self._init_unpickled_params()

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


class Input(Layer):

    def __init__(self, frozen=False):
        super(Input, self).__init__(frozen=frozen)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.output = x
        return self.output

    def backward(self, dvals: np.ndarray):
        return dvals

    def is_trainable(self) -> bool:
        return False

    def __getstate__(self):
        return super(Input, self).__getstate__()

    def __setstate__(self, state):
        super(Input, self).__setstate__(state)


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

        # Set regularizer and its updates
        self.weights_regularizer = weights_regularizer
        self.biases_regularizer = biases_regularizer

        # Set unpickled params
        self._init_unpickled_params()

    def _init_unpickled_params(self):
        super(Linear, self)._init_unpickled_params()
        # Set updates for backward
        self.dweights = None
        self.dbiases = None
        # Set updates for momentums
        self.weight_momentums = np.zeros_like(self.weights)
        self.bias_momentums = np.zeros_like(self.biases)
        self.weights_reg_updates = None
        self.biases_reg_updates = None

    def is_trainable(self) -> bool:
        return not self.frozen

    def get_weights(self, copy=True) -> np.ndarray:
        return self.weights.copy() if copy else self.weights

    def get_biases(self, copy=True) -> np.ndarray | None:
        return self.biases.copy() if copy else self.biases

    def get_dweights(self, copy=True) -> np.ndarray:
        return self.dweights.copy() if copy else self.dweights

    def get_dbiases(self, copy=True) -> np.ndarray | None:
        return self.dbiases.copy() if copy else self.dbiases

    def get_parameters(self):
        return {
            'weights': self.weights,
            'biases': self.biases,
        }

    def __getstate__(self):
        state = super(Linear, self).__getstate__()
        state.update({
            'weights': self.weights,
            'biases': self.biases,
            'in_features': self.in_features,
            'out_features': self.out_features,
            'grad_reduction': self.grad_reduction,
            'weights_regularizer': self.weights_regularizer,    # todo getstate?
            'biases_regularizer': self.biases_regularizer,      # todo getstate?
        })
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        super(Linear, self).__setstate__(state)

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

    # noinspection PyAttributeOutsideInit
    def backward(self, dvals: np.ndarray):
        # We don't want to compute updates if we are not training the model!
        if self.is_training():
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
    def is_trainable(self) -> bool:
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

    def __getstate__(self):
        state = super(ReLU, self).__getstate__()
        state.update({
            'subgrad_func': self.subgrad_func,
        })
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        super(ReLU, self).__setstate__(state)

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

    def __getstate__(self):
        state = super(SoftmaxLayer, self).__getstate__()
        state.update({
            'softmax': self.softmax,
        })
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        super(SoftmaxLayer, self).__setstate__(state)

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

    # Methods below (set_to_train, ..., unfreeze_layer) ensure that training and freeze state
    # is maintained consistently with the underlying linear and activation layer

    def set_to_train(self):
        self.linear.set_to_train()
        self.activation.set_to_train()
        super(Dense, self).set_to_train()

    def set_to_eval(self):
        self.linear.set_to_eval()
        self.activation.set_to_eval()
        super(Dense, self).set_to_eval()

    def freeze_layer(self):
        self.linear.freeze_layer()
        self.activation.freeze_layer()
        super(Dense, self).freeze_layer()

    def unfreeze_layer(self):
        self.linear.unfreeze_layer()
        self.activation.unfreeze_layer()
        super(Dense, self).unfreeze_layer()

    def _init_unpickled_params(self):
        super(Dense, self)._init_unpickled_params()
        self.net = None

    def __getstate__(self):
        state = super(Dense, self).__getstate__()
        state.update({
            'linear': self.linear,
            'activation': self.activation,
        })
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        super(Dense, self).__setstate__(state)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.net = self.linear.forward(x)
        self.output = self.activation.forward(self.net)
        return self.output

    def backward(self, dvals: np.ndarray):
        dvals = self.activation.backward(dvals)
        return self.linear.backward(dvals)

    def is_trainable(self) -> bool:
        return self.linear.is_trainable() and not self.frozen

    def get_parameters(self) -> dict:
        return {
            'linear': self.linear.get_parameters(),
            'activation': self.activation.get_parameters(),
        }


__all__ = [
    'Layer',
    'Input',
    'Linear',
    'Activation',
    'Dense',
    'Sigmoid',
    'Tanh',
    'ReLU',
    'SoftmaxLayer',
]
