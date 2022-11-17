"""
Basic functions with diffs and vjp registrations.
# todo VJPs in this case produce (l, 1, n) outputs.
"""
from __future__ import annotations
from .utils import *
import core.diffs as dfs
from core.diffs import row_vector_input_shape_checker as rv_input_check,\
    row_vector_input_shape_normalizer as rv_input_norm


class Softmax(Callable):

    def __init__(self, const_shift=0, max_shift=False):
        super(Softmax, self).__init__()
        self.const_shift = const_shift  # constant shift for arguments
        self.max_shift = max_shift      # subtracting the maximum input value from all inputs

    @dfs.set_primitive(input_checker=rv_input_check, input_normalizer=rv_input_norm, input_arg=1)
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.const_shift != 0:
            x += self.const_shift
        if self.max_shift:
            # Get unnormalized probabilities
            exp_values = np.exp(x - np.max(x, axis=-1, keepdims=True))
        else:
            exp_values = np.exp(x)
        s = np.sum(exp_values, axis=-1, keepdims=True)
        return exp_values / s

    @dfs.set_jacobian(__call__)
    def jacobian(self, x: np.ndarray):
        s = self(x)  # (l, 1, n)
        # Calculate the common part
        st = np.transpose(s, axes=[0, -1, -2])  # (l, n, 1)
        y = (-1.0) * (st @ s)  # (l, n, n)
        # Reshape for summing elements on the ("last two") diagonals
        l, n = y.shape[0], y.shape[2]
        ind = np.array(range(n))
        y[:, ind, ind] = s[:, 0, ind]
        return y

    @dfs.set_vjp(__call__)
    def vjp(self, x: np.ndarray, dvals: np.ndarray):
        s = self(x)
        sd = s * dvals
        sd_sum = np.sum(sd, axis=1, keepdims=True)  # todo check axis!
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
        y = self.coefficient * np.mean(np.sum((self.truth_values - x)**2, axis=-1), axis=0)
        return y

    @dfs.set_grad(__call__, input_checker=rv_input_check, input_normalizer=rv_input_norm, input_arg=1)
    def grad(self, x: np.ndarray):
        c = -2.0 * self.coefficient / self.truth_values.shape[0]
        return c * (self.truth_values - x)


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


__all__ = [
    'dfs',  # conveniently import module with "uniform" name
    'Softmax',
    'CategoricalCrossEntropy',
    'SquareError',
    'MeanSquareError',
    'softmax',
    'cross_entropy',
]
