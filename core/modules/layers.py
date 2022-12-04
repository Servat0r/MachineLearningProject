# Base layers for a Neural Network
from __future__ import annotations

import numpy as np

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
        self._training = None
        self.input = None
        self.output = None
        self._serialize_all = None

    def equals(self, other, include_updates=False, include_all=False):
        """
        Base utility method to check equality between two layers by discriminating
            1) case when updates are not of concern (e.g. after training);
            2) case when updates are of concern but input/output not;
            3) case when both updates and input/output are of concern (e.g. a backup).

        todo actually it does NOT work properly for input / output (now they are not saved)

        :param other: Other layer for verifying equality.
        :param include_updates: For case 2) (used in subclasses).
        :param include_all: For case 3) (used in subclasses).
        """
        # include_all overwrites include_updates
        if not isinstance(other, type(self)):
            return False
        check = all([
            self.frozen == other.frozen, self._training == other._training
        ])
        return check

    def _init_unpickled_parameters(self):
        """
        Re-initialization of parameters that are not serialized by Model.save().
        """
        self.input = None
        self.output = None

    def freeze_layer(self):
        self.frozen = True

    def unfreeze_layer(self):
        self.frozen = False

    def set_to_train(self):
        self._training = True

    def set_to_eval(self):
        self._training = False

    def set_serialize_all(self):
        self._serialize_all = True

    def unset_serialize_all(self):
        self._serialize_all = False

    def is_training(self):
        return self.is_trainable() and self._training

    @abstractmethod
    def is_trainable(self) -> bool:
        """
        Returns True iff layers contains any parameter (weights, biases) and is NOT frozen.
        """
        pass

    def get_parameters(self, copy=False) -> dict:
        """
        Retrieves layer parameters as a dictionary ('weights', 'biases', 'delta_weights' etc.).
        """
        return {}   # By default, no parameters for this layer

    def set_parameters(self, parameter_dict: dict):
        """
        Sets layer parameters from a given dictionary, whose syntax **must** respect that
        of get_parameters().
        """
        if not isinstance(parameter_dict, dict):
            raise TypeError(f"Wrong type {type(parameter_dict)} when setting parameters for {self}")
        for parameter_name, parameter_value in parameter_dict.items():
            if isinstance(parameter_value, np.ndarray):
                setattr(self, parameter_name, parameter_value)
            elif isinstance(parameter_value, dict):
                attribute = getattr(self, parameter_name)
                if attribute is None:
                    raise AttributeError(f"Unknown attribute {parameter_name} for {type(self)}")
                elif not isinstance(attribute, Layer):
                    raise TypeError(f"{attribute} is not an instance of {Layer}")
                else:
                    attribute.set_parameters(parameter_value)
            else:
                raise TypeError(f"Wrong type {type(parameter_value)} for {parameter_name} when parsing parameters dict")

    def __getstate__(self):
        state = {
            'frozen': self.frozen,
            '_training': self._training,
            '_serialize_all': self._serialize_all,
        }
        return state

    def __setstate__(self, state):
        frozen = state.pop('frozen', False)
        is_training = state.pop('_training', 2)  # neither True, False nor None
        if is_training == 2:
            raise RuntimeError('Attribute \'_is_training\' not available!')
        serialize_all = state.pop('_serialize_all', 2)
        if serialize_all == 2:
            raise RuntimeError('Attribute \'_serialize_all\' not available!')
        self.frozen = frozen
        self._training = is_training
        self._serialize_all = serialize_all
        self._init_unpickled_parameters()

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the current level.
        :param x: Input values as (l, 1, n) numpy array
        (l = number of examples, n = input dimension).
        """
        pass

    @abstractmethod
    def backward(self, delta_vals: np.ndarray):
        """
        Backward pass of the current level.
        :param delta_vals: "Differential" values of the successive layer
        (in terms of the computational graph associated to the network).
        :return "Differential" values for the previous layer.
        """
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

    def backward(self, delta_vals: np.ndarray):
        return delta_vals

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
                 biases_initializer: Initializer = ZeroInitializer(), gradients_reduction='mean', frozen=False,
                 weights_regularizer: Regularizer = None, biases_regularizer: Regularizer = None, dtype=np.float64):
        """
        :param in_features: Input dimension.
        :param out_features: Output dimension.
        :param weights_initializer: Initializer to use for weights initialization.
        :param biases_initializer: Initializer to use for biases initialization. Defaults
        to a ZeroInitializer.
        :param gradients_reduction: Reduction to apply to the gradients of a batch of
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
        self.gradients_reduction = gradients_reduction
        self.dtype = dtype

        # Set regularizer and its updates
        self.weights_regularizer = weights_regularizer
        self.biases_regularizer = biases_regularizer

        # Set updates for backward
        self.delta_weights = None
        self.delta_biases = None
        # Set updates for momentums
        self.weight_momentums = np.zeros_like(self.weights)
        self.bias_momentums = np.zeros_like(self.biases)
        self.weights_regularization_updates = None
        self.biases_regularization_updates = None

    def equals(self, other, include_updates=False, include_all=False):
        check = super(Linear, self).equals(other, include_updates, include_all)
        if not check or not isinstance(other, Linear):
            return False
        # Case 3) includes Case 2)
        include_updates = True if include_all else include_updates
        # Base checks (Case 1)
        check = [
            self.in_features == other.in_features, self.out_features == other.out_features,
            np.equal(self.weights, other.weights).all(), np.equal(self.biases, other.biases).all(),
            self.gradients_reduction == other.gradients_reduction, self.dtype == other.dtype,
            self.weights_regularizer == other.weights_regularizer,
            self.biases_regularizer == other.biases_regularizer,
        ]
        if not all(check):
            return False
        if include_updates:
            # Checks for Case 2)
            check = [
                np.equal(self.delta_weights, other.delta_weights).all(), np.equal(self.delta_biases, other.delta_biases).all(),
                np.equal(self.weight_momentums, other.weight_momentums).all(),
                np.equal(self.bias_momentums, other.bias_momentums).all(),
                np.equal(self.weights_regularization_updates, other.weights_regularization_updates).all(),
                np.equal(self.biases_regularization_updates, other.biases_regularization_updates).all(),
            ]
            if not all(check):
                return False
        return True

    def _init_unpickled_parameters(self):
        super(Linear, self)._init_unpickled_parameters()
        if not self.is_training() and not self._serialize_all:
            # Set updates for backward
            self.delta_weights = None
            self.delta_biases = None
            # Set updates for momentums
            self.weight_momentums = np.zeros_like(self.weights)
            self.bias_momentums = np.zeros_like(self.biases)
            self.weights_regularization_updates = None
            self.biases_regularization_updates = None

    def is_trainable(self) -> bool:
        return not self.frozen

    def get_weights(self, copy=True) -> np.ndarray:
        return self.weights.copy() if copy else self.weights

    def get_biases(self, copy=True) -> np.ndarray | None:
        return self.biases.copy() if copy else self.biases

    def get_delta_weights(self, copy=True) -> np.ndarray:
        return self.delta_weights.copy() if copy else self.delta_weights

    def get_delta_biases(self, copy=True) -> np.ndarray | None:
        return self.delta_biases.copy() if copy else self.delta_biases

    def get_parameters(self, copy=False):
        return {
            'weights': self.get_weights(copy=copy),
            'biases': self.get_biases(copy=copy),
        }

    def __getstate__(self):
        state = super(Linear, self).__getstate__()
        # Attributes that are ALWAYS saved (Case 1)
        state.update({
            'weights': self.weights,
            'biases': self.biases,
            'in_features': self.in_features,
            'out_features': self.out_features,
            'gradients_reduction': self.gradients_reduction,
            'dtype': self.dtype,
            'weights_regularizer': self.weights_regularizer,    # todo getstate?
            'biases_regularizer': self.biases_regularizer,      # todo getstate?
        })
        if self.is_training() or self._serialize_all:
            # Attributes that are saved ONLY if model is layer is in
            # a training loop or if requested by the caller (Cases 2-3)
            state.update({
                'delta_weights': self.delta_weights,
                'delta_biases': self.delta_biases,
                'weight_momentums': self.weight_momentums,
                'bias_momentums': self.bias_momentums,
                'weights_regularization_updates': self.weights_regularization_updates,
                'biases_regularization_updates': self.biases_regularization_updates,
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
            return np.expand_dims(x, axes=[1, 2])  # todo check if it overlows with axes (not clear from numpy doc)
        elif len(shape) == 2:
            return np.expand_dims(x, axis=1)
        else:
            return x

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = self.normalize_input_shape(x)
        self.output = self.input @ self.weights + self.biases
        return self.output

    def backward(self, delta_vals: np.ndarray):
        # We don't want to compute updates if we are not training the model!
        if self.is_training():
            transposed_input = np.transpose(self.input, axes=[0, 2, 1])  # (l, m, 1)
            # Calculate update to layer's weights and biases
            self.delta_weights = transposed_input @ delta_vals  # matmul operator
            self.delta_biases = delta_vals.copy()

            # Apply reductions if requested
            if self.gradients_reduction == 'sum':
                self.delta_weights = np.sum(self.delta_weights, axis=0)
                self.delta_biases = np.sum(self.delta_biases, axis=0)
            elif self.gradients_reduction == 'mean':
                self.delta_weights = np.mean(self.delta_weights, axis=0)
                self.delta_biases = np.mean(self.delta_biases, axis=0)

            # Handle regularization
            if self.weights_regularizer is not None:
                self.weights_regularization_updates = self.weights_regularizer.update(self.weights)

            if self.biases_regularizer is not None:
                self.biases_regularization_updates = self.biases_regularizer.update(self.biases)

        # Now calculate values to backpropagate to previous layer
        return np.dot(delta_vals, self.weights.T)


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

    def backward(self, delta_vals: np.ndarray):
        return delta_vals * self.output * (1 - self.output)


class Tanh(Activation):

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.output = np.tanh(x)
        return self.output

    def backward(self, delta_vals: np.ndarray):
        return delta_vals * (1. - np.square(self.output))


class ReLU(Activation):

    def default_subgradient_func(self, delta_values: np.ndarray):
        """
        Default subgradient for ReLU: for each input component,
        1 iff it is > 0, else 0; this is then multiplied by the
        delta_values input to get out_delta_values.
        """
        out_delta_values = delta_values.copy()
        out_delta_values[self.input <= 0] = 0
        return out_delta_values

    def __init__(self, frozen=False, subgrad_func: Callable = None):
        super(ReLU, self).__init__(frozen=frozen)
        self.subgrad_func = self.default_subgradient_func if subgrad_func is None else subgrad_func

    def __eq__(self, other):
        return super(ReLU, self).__eq__(other) and self.subgrad_func == other.subgrad_func

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

    def backward(self, delta_vals: np.ndarray):
        return self.subgrad_func(delta_vals)


class SoftmaxLayer(Activation):

    def __init__(self, const_shift=0, max_shift=False):
        super(SoftmaxLayer, self).__init__()
        self.softmax = cf.Softmax(const_shift=const_shift, max_shift=max_shift)

    def equals(self, other, include_updates=False, include_all=False):
        return super(SoftmaxLayer, self).equals(other, include_updates, include_all) and self.softmax == other.softmax

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

    def backward(self, delta_vals: np.ndarray):
        return self.softmax.vjp(self.input, delta_vals)


class Dense(Layer):
    """
    A fully-connected layer with activation function for all the units.
    """
    def __init__(
            self, in_features: int, out_features: int, activation_layer: Activation,
            weights_initializer: Initializer, biases_initializer: Initializer = ZeroInitializer(),
            gradients_reduction='mean', frozen=False, weights_regularizer: Regularizer = None,
            biases_regularizer: Regularizer = None, dtype=np.float64
    ):
        super(Dense, self).__init__(frozen=frozen)
        # Initialize linear part
        self.linear = Linear(
            in_features, out_features, weights_initializer, biases_initializer,
            gradients_reduction, weights_regularizer=weights_regularizer,
            biases_regularizer=biases_regularizer, dtype=dtype,
        )
        self.activation = activation_layer
        self.net = None

    def equals(self, other, include_updates=False, include_all=False):
        if not super(Dense, self).equals(other, include_updates, include_all):
            return False
        check = [
            self.linear.equals(other.linear, include_updates, include_all),
            self.activation.equals(other.activation, include_updates, include_all),
        ]
        if not all(check):
            return False
        if include_updates:
            # self.net actually contains an intermediate input
            return np.equal(self.net, other.net).all()
        return True

    # Methods below (set_serialize_all, ..., unfreeze_layer) ensure that training, serialization
    # and freeze state is maintained consistently with the underlying linear and activation layer

    def set_serialize_all(self):
        self.linear.set_serialize_all()
        self.activation.set_serialize_all()
        super(Dense, self).set_serialize_all()

    def unset_serialize_all(self):
        self.linear.unset_serialize_all()
        self.activation.unset_serialize_all()
        super(Dense, self).unset_serialize_all()

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

    def _init_unpickled_parameters(self):
        super(Dense, self)._init_unpickled_parameters()
        if not self.is_training() and not self._serialize_all:
            self.net = None

    def __getstate__(self):
        state = super(Dense, self).__getstate__()
        state.update({
            'linear': self.linear,
            'activation': self.activation,
        })
        if self.is_training() or self._serialize_all:
            state.update({
                'net': self.net,
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

    def backward(self, delta_vals: np.ndarray):
        delta_vals = self.activation.backward(delta_vals)
        return self.linear.backward(delta_vals)

    def is_trainable(self) -> bool:
        return self.linear.is_trainable() and not self.frozen

    def get_parameters(self, copy=False) -> dict:
        return {
            'linear': self.linear.get_parameters(copy=copy),
            'activation': self.activation.get_parameters(copy=copy),
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
