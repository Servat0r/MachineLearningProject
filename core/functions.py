"""
Basic functions with diffs and vjp registrations.
# todo VJPs in this case produce (l, 1, n) outputs.
"""
from __future__ import annotations
from .utils import *
import core.diffs as dfs
from core.diffs import row_vector_input_shape_checker as rv_input_check, row_vector_input_shape_normalizer as rv_input_norm


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
    return _relu_egrad(x) * dvals


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
def square_error(x: np.ndarray, const: float = 0.5):
    y = const * np.square(np.linalg.norm(x, axis=2))
    return y


@dfs.set_grad(square_error, input_checker=rv_input_check, input_normalizer=rv_input_norm)
def _square_error_grad(x: np.ndarray, const: float = 0.5):
    return 2 * const * x


@dfs.set_primitive(input_checker=rv_input_check, input_normalizer=rv_input_norm)
def abs_error(x: np.ndarray):
    y = np.sum(np.abs(x), axis=2)
    return y


__all__ = [
    'sigmoid',
    'sign',
    'tanh',
    'relu',
    'softmax',
    'cross_entropy',
    'square_error',
    'abs_error',
]
