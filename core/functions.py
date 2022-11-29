from __future__ import annotations
from .utils import *


class Softmax(Callable):

    def __init__(self, const_shift=0, max_shift=False):
        super(Softmax, self).__init__()
        self.const_shift = const_shift  # constant shift for arguments
        self.max_shift = max_shift      # subtracting the maximum input value from all inputs

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
        st = np.transpose(s, axes=[0, -1, -2])  # (l, n, 1)
        y = (-1.0) * (st @ s)  # (l, n, n)
        # Reshape for summing elements on the ("last two") diagonals
        l, n = y.shape[0], y.shape[2]
        ind = np.array(range(n))
        y[:, ind, ind] = s[:, 0, ind]
        return y

    def vjp(self, x: np.ndarray, dvals: np.ndarray):
        s = self(x)
        sd = s * dvals
        sd_sum = np.sum(sd, axis=1, keepdims=True)  # todo check axis!
        return sd - sd_sum * s


class CategoricalCrossEntropy(Callable):

    def __init__(self, clip_value: TReal = 1e-7):
        super(CategoricalCrossEntropy, self).__init__()
        self.clip_value = clip_value

    def __call__(self, x: np.ndarray, truth: np.ndarray) -> np.ndarray:
        samples = len(x)
        x_clipped = np.clip(x, self.clip_value, 1 - self.clip_value)
        correct_confidences = []
        trshape = truth.shape
        if len(trshape) == 1:  # (l,)
            correct_confidences = x_clipped[range(samples), 0, truth]
        elif len(trshape) == 3:  # (l, 1, n)
            filtered_x = x_clipped * truth
            correct_confidences = np.sum(filtered_x, axis=2)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def grad(self, x: np.ndarray, truth: np.ndarray):
        x_clip = - 1.0 / np.clip(x, self.clip_value, 1 - self.clip_value)   # todo maybe this clip should not occur
        trshape = truth.shape
        if len(trshape) == 1:
            # If labels are sparse, turn them into one-hot encoded ones
            truth = np.eye(x.shape[2])[truth]
            truth = np.reshape(truth, (trshape[0], 1, x.shape[2]))
        result = x_clip * truth
        return result


class SquaredError(Callable):

    def __init__(self, const=0.5):
        self.const = const

    def __call__(self, x: np.ndarray, truth: np.ndarray):
        return self.const * np.sum(np.square(truth - x), axis=-1)

    def grad(self, x: np.ndarray, truth: np.ndarray):
        return -2 * self.const * (truth - x)


def accuracy(pred: np.ndarray, truth: np.ndarray, dtype=np.int32) -> np.ndarray:
    """
    Accuracy for arrays of scalar values.
    """
    return np.equal(pred, truth).astype(dtype)


# todo this implementation shall be made more efficient (need to replace argmax for truth indexes
# todo with something that access the correct elements of the array and return the array given by
# todo accessing with them)
def categorical_accuracy(pred: np.ndarray, truth: np.ndarray, dtype=np.int32) -> np.ndarray:
    """
    Accuracy for one-hot encoded labels.
    """
    pred_indexes = np.argmax(pred, axis=-1)
    truth_indexes = np.argmax(truth, axis=-1)
    return np.equal(pred_indexes, truth_indexes).astype(dtype)


def binary_accuracy(pred: np.ndarray, truth: np.ndarray, threshold=0.5, dtype=np.int32):
    """
    Accuracy for binary labels. Predicted labels are treated
    as ones if above threshold and zeros otherwise.
    """
    pred_transformed = np.zeros_like(pred, dtype=truth.dtype)
    pred_transformed[pred >= threshold] = 1.
    return accuracy(pred_transformed, truth, dtype)


def mean_euclidean_error(pred: np.ndarray, truth: np.ndarray, reduce=True, dtype=np.float64):
    """
    Mean Euclidean Error, i.e. average over all examples of 2-norm of that example.
    :param pred: Predicted values.
    :param truth: Ground truth values.
    :param reduce: If True, average over examples will be calculated; otherwise,
    raw values for each example will be returned (see MeanEuclideanError metric).
    :param dtype: Data type of result array. Defaults to np.float64.
    """
    raw_values = np.linalg.norm(pred - truth, axis=-1)
    if reduce:
        return raw_values.astype(dtype)
    else:
        return np.mean(raw_values, dtype=dtype)


def root_mean_squared_error(pred: np.ndarray, truth: np.ndarray, dtype=np.float64):
    """
    Root Mean Squared Error, i.e. square root of the average
    of all squared 2-norms of the examples.
    """
    norms = np.sum(np.square(pred - truth), axis=-1)
    return np.sqrt(np.mean(norms, dtype=dtype))


__all__ = [
    'Softmax',
    'CategoricalCrossEntropy',
    'SquaredError',
    'accuracy',
    'categorical_accuracy',
    'binary_accuracy',
    'mean_euclidean_error',
    'root_mean_squared_error',
]
