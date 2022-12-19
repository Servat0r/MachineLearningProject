from __future__ import annotations
from .utils import *


class Softmax(Callable):

    def __init__(self, const_shift=0, max_shift=False):
        super(Softmax, self).__init__()
        self.const_shift = const_shift  # constant shift for arguments
        self.max_shift = max_shift      # subtracting the maximum input value from all inputs

    def __eq__(self, other):
        if not super(Softmax, self).__eq__(other) or not isinstance(other, Softmax):
            return False
        return all([self.const_shift == other.const_shift, self.max_shift == other.max_shift])

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

    def jacobian(self, x: np.ndarray):
        s = self(x)  # (l, 1, n)
        # Calculate the common part
        s_transposed = np.transpose(s, axes=[0, -1, -2])  # (l, n, 1)
        y = (-1.0) * (s_transposed @ s)  # (l, n, n)
        # Reshape for summing elements on the ("last two") diagonals
        l, n = y.shape[0], y.shape[2]
        ind = np.array(range(n))
        y[:, ind, ind] += s[:, 0, ind]
        return y

    def vjp(self, x: np.ndarray, delta_values: np.ndarray):    # todo check!
        s = self(x)
        sd = s * delta_values
        sd_sum = np.sum(sd, axis=-1, keepdims=True)
        return sd - sd_sum * s


class CategoricalCrossEntropy(Callable):

    def __init__(self, clip_value: TReal = 1e-7, clip=False):
        super(CategoricalCrossEntropy, self).__init__()
        self.clip_value = clip_value
        self.clip = clip

    def __eq__(self, other):
        if not super(CategoricalCrossEntropy, self).__eq__(other) or not isinstance(other, CategoricalCrossEntropy):
            return False
        return self.clip_value == other.clip_value

    def __call__(self, x: np.ndarray, truth: np.ndarray) -> np.ndarray:
        samples = len(x)
        if self.clip:
            x_clipped = np.clip(x, self.clip_value, 1 - self.clip_value)
        else:
            x_clipped = x
        correct_confidences = []
        trshape = truth.shape
        if len(trshape) == 1:  # (l,)
            correct_confidences = x_clipped[range(samples), truth]
        elif len(trshape) == 2:  # (l, n)
            filtered_x = x_clipped * truth
            correct_confidences = np.sum(filtered_x, axis=-1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def grad(self, x: np.ndarray, truth: np.ndarray):
        x_clip = - 1.0 / x
        trshape = truth.shape
        if len(trshape) == 1:
            # If labels are sparse, turn them into one-hot encoded ones
            ohe_truth = np.zeros_like(x_clip)  # (l, 1, n)
            for i in range(trshape[0]):
                ohe_truth[i, 0, truth[i]] = 1.
            truth = ohe_truth
        result = x_clip * truth
        return result


class SquaredError(Callable):

    def __init__(self, const=1.0):
        self.const = const

    def __eq__(self, other):
        if not super(SquaredError, self).__eq__(other) or not isinstance(other, SquaredError):
            return False
        return self.const == other.const

    def __call__(self, x: np.ndarray, truth: np.ndarray):
        return self.const * np.sum(np.square(truth - x), axis=-1)

    def grad(self, x: np.ndarray, truth: np.ndarray):
        return -2 * self.const * (truth - x)


class MeanSquaredError(SquaredError):

    def __call__(self, x: np.ndarray, truth: np.ndarray):
        return self.const * np.mean(np.square(truth - x), axis=-1)

    def grad(self, x: np.ndarray, truth: np.ndarray):
        return -2 * self.const * (truth - x) / truth.shape[-1]


def accuracy(predicted: np.ndarray, truth: np.ndarray, dtype=np.int32) -> np.ndarray:
    """
    Accuracy for arrays of scalar values.
    """
    return np.equal(predicted, truth).astype(dtype)


# todo this implementation shall be made more efficient (need to replace argmax for truth indexes
# todo with something that access the correct elements of the array and return the array given by
# todo accessing with them)
def categorical_accuracy(predicted: np.ndarray, truth: np.ndarray, dtype=np.int32) -> np.ndarray:
    """
    Accuracy for one-hot encoded labels.
    """
    predicted_indexes = np.argmax(predicted, axis=-1)
    truth_indexes = np.argmax(truth, axis=-1)
    return np.equal(predicted_indexes, truth_indexes).astype(dtype)


def sparse_categorical_accuracy(predicted: np.ndarray, truth: np.ndarray, dtype=np.int32) -> np.ndarray:
    """
    Accuracy for integer labels.
    """
    predicted_indexes = np.argmax(predicted, axis=-1).astype(np.int).reshape(truth.shape)
    return np.equal(predicted_indexes, truth).astype(dtype)


def binary_accuracy(predicted: np.ndarray, truth: np.ndarray, threshold=0.5, dtype=np.int32):
    """
    Accuracy for binary labels. Predicted labels are treated
    as ones if above threshold and zeros otherwise.
    """
    predicted_transformed = np.zeros_like(predicted, dtype=truth.dtype)
    predicted_transformed[predicted >= threshold] = 1.
    return accuracy(predicted_transformed, truth, dtype)


def mean_absolute_error(predicted: np.ndarray, truth: np.ndarray, reduce=True, dtype=np.float32):
    raw_values = np.mean(np.abs(truth - predicted), axis=-1)
    if reduce:
        return np.mean(raw_values, dtype=dtype)
    else:
        return raw_values.astype(dtype)


def mean_euclidean_error(predicted: np.ndarray, truth: np.ndarray, reduce=True, dtype=np.float32):
    """
    Mean Euclidean Error, i.e. average over all examples of 2-norm of that example.
    :param predicted: Predicted values.
    :param truth: Ground truth values.
    :param reduce: If True, average over examples will be calculated; otherwise,
    raw values for each example will be returned (see MeanEuclideanError metric).
    :param dtype: Data type of result array. Defaults to np.float32.
    """
    raw_values = np.linalg.norm(predicted - truth, axis=-1)
    if reduce:
        return np.mean(raw_values, dtype=dtype)
    else:
        return raw_values.astype(dtype)


def root_mean_squared_error(predicted: np.ndarray, truth: np.ndarray, dtype=np.float32):
    """
    Root Mean Squared Error, i.e. square root of the average
    of all squared 2-norms of the examples.
    """
    norms = np.sum(np.square(predicted - truth), axis=-1)
    return np.sqrt(np.mean(norms, dtype=dtype))


__all__ = [
    'Softmax',
    'CategoricalCrossEntropy',
    'SquaredError',
    'MeanSquaredError',
    'accuracy',
    'categorical_accuracy',
    'sparse_categorical_accuracy',
    'binary_accuracy',
    'mean_absolute_error',
    'mean_euclidean_error',
    'root_mean_squared_error',
]
