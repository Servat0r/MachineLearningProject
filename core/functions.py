"""
Basic functions with diffs and vjp registrations.
# todo VJPs in this case produce (l, 1, n) outputs.
"""
from __future__ import annotations
from .utils import *
import core.diffs as dfs
from core.diffs import row_vector_input_shape_checker as rv_input_check,\
    row_vector_input_shape_normalizer as rv_input_norm


# Class-based activation functions
class ActivationFunction(Callable):
    """
    Base class for activation functions.
    """
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass


class Softmax(Callable):

    def __init__(self, const_shift=0, max_shift=False):
        super(Softmax, self).__init__()
        self.const_shift = const_shift  # constant shift for arguments
        self.max_shift = max_shift      # subtracting the maximum input value from all inputs

    def __shift_x(self, x: np.ndarray) -> np.ndarray:
        """
        Applies constant and/or maximum shift to input data BEFORE applying Softmax.
        This is a common technique for avoiding numerical instability if the input
        array contains large positive values. Since the function x -> x - c for c
        constant has the identity matrix as differential, the operation is made
        in-place since the backprop value is 1.  # todo check if it is correct!
        """
        if self.const_shift != 0:
            x += self.const_shift
        if self.max_shift:
            x -= np.max(x, axis=2, keepdims=True)
        return x

    @dfs.set_primitive(input_checker=rv_input_check, input_normalizer=rv_input_norm, input_arg=1)
    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.__shift_x(x)   # todo check if actually modifies inplace!
        xc = np.exp(x)
        s = np.sum(xc, axis=2, keepdims=True)
        xc /= s
        return xc

    @dfs.set_jacobian(__call__)
    def jacobian(self, x: np.ndarray):
        s = self(x)  # (l, 1, n)
        # Calculate the common part
        st = np.transpose(s, axes=[0, 2, 1])  # (l, n, 1)
        y = (-1.0) * (st @ s)  # (l, n, n)
        # Reshape for summing elements on the ("last two") diagonals
        l, n = y.shape[0], y.shape[2]
        ind = np.array(range(n))
        y[:, ind, ind] = s[:, 0, ind]
        # todo actually for small matrices (10^1 or 10^2) a double for-loop is faster
        # todo for 10^3 or higher, the above y[:,ind,ind] ... is much faster
        return y

    @dfs.set_vjp(__call__)
    def vjp(self, x: np.ndarray, dvals: np.ndarray):
        s = self(x)
        sd = s * dvals
        sd_sum = np.sum(sd, axis=1, keepdims=True)
        return sd - sd_sum * s


class CategoricalCrossEntropy(Callable):

    def __init__(self, target_truth_values: np.ndarray, clip_value: TReal = 1e-7):
        super(CategoricalCrossEntropy, self).__init__()
        target_truth_values = dfs.row_vector_target_shape_normalizer(target_truth_values)
        self.target_truth_values = target_truth_values
        self.clip_value = clip_value

    def set_truth_values(self, truth_values: np.ndarray):
        self.target_truth_values = truth_values

    def check_input_shape(self, x: np.ndarray) -> bool:
        """
        Input values should have the shape (l, 1, n).
        :param x:
        :return:
        """
        x_shape = x.shape
        t_shape = self.target_truth_values.shape
        return all([
            len(x_shape) == 3, x_shape[0] == t_shape[0], x_shape[1] == 1,
            x_shape[2] > 0 if len(t_shape) == 1 else x_shape[2] == t_shape[2],  # todo check what's going on here!
        ])

    @dfs.set_primitive(input_checker=check_input_shape, input_normalizer=rv_input_norm, input_arg=1)
    def __call__(self, x: np.ndarray):
        samples = len(x)
        x_clipped = np.clip(x, self.clip_value, 1 - self.clip_value)
        correct_confidences = []

        trshape = self.target_truth_values.shape
        if len(trshape) == 1:  # (l,)
            correct_confidences = x_clipped[range(samples), 0, self.target_truth_values]
        elif len(trshape) == 3:  # (l, 1, n)
            filtered_x = x_clipped * self.target_truth_values
            correct_confidences = np.sum(filtered_x, axis=2)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    @dfs.set_grad(__call__, input_checker=check_input_shape, input_normalizer=rv_input_norm, input_arg=1)
    def grad(self, x: np.ndarray):
        x_clip = - 1.0 / np.clip(x, self.clip_value, 1 - self.clip_value)
        trshape = self.target_truth_values.shape
        if len(trshape) == 1:
            # If labels are sparse, turn them into one-hot encoded ones
            self.target_truth_values = np.eye(x.shape[2])[self.target_truth_values]
            self.target_truth_values = np.reshape(self.target_truth_values, (trshape[0], 1, x.shape[2]))
        result = x_clip * self.target_truth_values
        return result


# Generic square error
class SquareError(Callable):
    """
    Calculates square error (coefficient * ||x||^2) for the patterns SEPARATELY.
    """
    def __init__(self, truth_values: np.ndarray, coefficient=0.5):
        self.coefficient = coefficient
        self.truth_values = truth_values

    @dfs.set_primitive(input_checker=rv_input_check, input_normalizer=rv_input_norm, input_arg=1)
    def __call__(self, x: np.ndarray):
        y = self.coefficient * np.square(np.linalg.norm(self.truth_values - x, axis=2))
        return y

    @dfs.set_grad(__call__, input_checker=rv_input_check, input_normalizer=rv_input_norm, input_arg=1)
    def grad(self, x: np.ndarray, coeff=1):
        return -1.0 * self.coefficient * coeff * 2 * x


# Mean Square Error over the first dimension
class MeanSquareError(Callable):
    """
    Calculates square error (coefficient * ||x||^2) for the patterns SEPARATELY.
    """
    def __init__(self, truth_values: np.ndarray = None, coefficient=0.5):
        self.coefficient = coefficient
        self.truth_values = truth_values

    @dfs.set_primitive(input_checker=rv_input_check, input_normalizer=rv_input_norm, input_arg=1)
    def __call__(self, x: np.ndarray):
        y = np.mean(self.coefficient * np.square(np.linalg.norm(self.truth_values - x, axis=2)), axis=0)
        return y

    @dfs.set_grad(__call__, input_checker=rv_input_check, input_normalizer=rv_input_norm, input_arg=1)
    def grad(self, x: np.ndarray):
        c = -2.0 * self.coefficient / self.truth_values.shape[0]
        return c * (self.truth_values - x)


# Sigmoid
@dfs.set_primitive(input_checker=rv_input_check, input_normalizer=rv_input_check)
def sigmoid(x: np.ndarray):
    return 1.0 / (1.0 + np.exp(-x))


# input check and normalization is done by sigmoid() call
@dfs.set_egrad(sigmoid)
def _sigmoid_egrad(x: np.ndarray):
    s = sigmoid(x)
    return s * (1.0 - s)


@dfs.set_vjp(sigmoid)
def _sigmoid_vjp(x: np.ndarray, dvals: np.ndarray):
    return _sigmoid_egrad(x) * dvals


# Sign
@dfs.set_primitive(input_checker=rv_input_check, input_normalizer=rv_input_norm)
def sign(x: np.ndarray):
    return np.sign(x)


# Tanh
@dfs.set_primitive(input_checker=rv_input_check, input_normalizer=rv_input_norm)
def tanh(x: np.ndarray):
    return np.tanh(x)


@dfs.set_egrad(tanh)
def _tanh_egrad(x: np.ndarray):
    return 1.0 - np.square(tanh(x))


@dfs.set_vjp(tanh)
def _tanh_vjp(x: np.ndarray, dvals: np.ndarray):
    return _tanh_egrad(x) * dvals


# ReLU
@dfs.set_primitive(input_checker=rv_input_check, input_normalizer=rv_input_norm)
def relu(x: np.ndarray):
    return np.maximum(x, np.zeros_like(x))


@dfs.set_egrad(relu)
def _relu_egrad(x: np.ndarray):
    # todo this "gradient" is actually incorrect since ReLU is not differentiable in 0
    # todo see if it is necessary to change it as it is said in ML or CM
    y: np.ndarray = relu(x)
    np.place(y, y > 0, 1)
    return y


@dfs.set_vjp(relu)
def _relu_vjp(x: np.ndarray, dvals: np.ndarray):
    dvals2 = dvals.copy()
    dvals2[x <= 0] = 0
    return dvals2
    # return _relu_egrad(x) * dvals


# Softmax
def __softmax_shift_x(x: np.ndarray, const_shift=0, max_shift=False):
    if const_shift != 0:
        x += const_shift
    if max_shift:
        xmax = np.reshape(np.max(x, axis=2), x.shape[0])  # todo check!
        x[:] -= xmax[:]  # todo check if correctly broadcasts each element of xmax to each line!
    return x


@dfs.set_primitive(input_checker=rv_input_check, input_normalizer=rv_input_norm)
def softmax(x: np.ndarray, const_shift=0, max_shift=False):
    x = __softmax_shift_x(x, const_shift, max_shift)
    xc = np.exp(x)
    s = np.sum(xc, axis=2, keepdims=True)
    xc /= s
    return xc


@dfs.set_jacobian(softmax)
def _softmax_jac(x: np.ndarray, const_shift=0, max_shift=False):
    """
    Calculates the jacobian of Softmax in x.
    """
    s = softmax(x, const_shift, max_shift)  # (l, 1, n)
    # Calculate the common part
    st = np.transpose(s, axes=[0, 2, 1])  # (l, n, 1)
    y = (-1.0) * (st @ s)  # (l, n, n)
    # Reshape for summing elements on the ("last two") diagonals
    l, n = y.shape[0], y.shape[2]
    ind = np.array(range(n))
    y[:, ind, ind] = s[:, 0, ind]
    # todo actually for small matrices (10^1 or 10^2) a double for-loop is faster
    # todo for 10^3 or higher, the above y[:,ind,ind] ... is much faster
    return y


@dfs.set_vjp(softmax)
def _softmax_vjp(x: np.ndarray, dvals: np.ndarray, const_shift=0, max_shift=False):
    s = softmax(x, const_shift, max_shift)
    sd = s * dvals
    sd_sum = np.sum(sd, axis=1, keepdims=True)
    return sd - sd_sum * s


# (Categorical) Cross Entropy
@dfs.set_primitive(input_checker=rv_input_check, input_normalizer=rv_input_norm)
@dfs.set_input_shape_normalizer(dfs.row_vector_target_shape_normalizer, inarg=1)
@dfs.set_input_shape_checker(dfs.row_vector_input_shape_checker, inarg=1)
def cross_entropy(x: np.ndarray, truth_values: np.ndarray, clip_value: TReal = 1e-7):
    samples = len(x)
    x_clipped = np.clip(x, clip_value, 1 - clip_value)
    correct_confidences = []

    trshape = truth_values.shape
    if len(trshape) == 1:  # (l,)
        correct_confidences = x_clipped[range(samples), 0, truth_values]
    elif len(trshape) == 3:  # (l, 1, n)
        filtered_x = x_clipped * truth_values
        correct_confidences = np.sum(filtered_x, axis=2)

    negative_log_likelihoods = -np.log(correct_confidences)
    return negative_log_likelihoods


@dfs.set_grad(cross_entropy, input_checker=rv_input_check, input_normalizer=rv_input_norm)
@dfs.set_input_shape_normalizer(dfs.row_vector_target_shape_normalizer, inarg=1)
@dfs.set_input_shape_checker(dfs.row_vector_input_shape_checker, inarg=1)
def _cross_entropy_grad(x: np.ndarray, truth_values: np.ndarray, clip_value: TReal = 1e-7):
    x_clip = - 1.0 / np.clip(x, clip_value, 1 - clip_value)
    trshape = truth_values.shape
    if len(trshape) == 1:
        # If labels are sparse, turn them into one-hot encoded ones
        truth_values = np.eye(x.shape[2])[truth_values]
        truth_values = np.reshape(truth_values, (trshape[0], 1, x.shape[2]))
    result = x_clip * truth_values
    return result


@dfs.set_primitive(input_checker=rv_input_check, input_normalizer=rv_input_norm)
def square_error(x: np.ndarray, truth: np.ndarray, const: float = 0.5):
    y = const * np.square(np.linalg.norm(truth - x, axis=2))
    return y


@dfs.set_grad(square_error, input_checker=rv_input_check, input_normalizer=rv_input_norm)
def _square_error_grad(x: np.ndarray, truth: np.ndarray, const: float = 0.5):
    return 2 * const * (truth - x)


@dfs.set_primitive(input_checker=rv_input_check, input_normalizer=rv_input_norm)
def abs_error(x: np.ndarray):
    y = np.sum(np.abs(x), axis=2)
    return y


__all__ = [
    'dfs',  # conveniently import module with "uniform" name

    'ActivationFunction',
    'Softmax',
    'CategoricalCrossEntropy',
    'SquareError',
    'MeanSquareError',

    'sigmoid',
    'sign',
    'tanh',
    'relu',
    'softmax',
    'cross_entropy',
    'square_error',
    'abs_error',
]
