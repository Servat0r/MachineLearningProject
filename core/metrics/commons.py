# Commons metrics, wrapped around commons functions
from __future__ import annotations
import time
from core.utils.types import *
from core.functions import *
from .base import *


class LambdaFunctionMetric(FunctionMetric):
    """
    Utility class for a FunctionMetric based on a function which is defined with some
    more parameters than (predicted, truth), but such that they can be fixed once for
    the entire lifecycle (e.g. datatype).
    """

    @abstractmethod
    def lambda_function(self, predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
        pass

    def __init__(self, *args, **kwargs):
        super(LambdaFunctionMetric, self).__init__(*args, **kwargs)
        self.func = self.lambda_function

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        self_dict.pop('func')
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.func = self.lambda_function

    def __eq__(self, other):
        if not isinstance(other, LambdaFunctionMetric):
            return False
        if not isinstance(other, type(self)):
            return False
        self_dict, other_dict = self.__dict__.copy(), other.__dict__.copy()
        # self.func and other.func actually are bound methods rather than "pure" functions.
        # Without excluding them, == will always give False.
        # Since we check that self and other are of the same type, for these simple cases
        # this implies that also functions are the same, except for extra arguments.
        self_dict.pop('func')
        other_dict.pop('func')
        return self_dict == other_dict


class Accuracy(LambdaFunctionMetric):
    """
    Accuracy metric for numerical (scalar) labels.
    """
    def lambda_function(self, predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
        return accuracy(predicted, truth, dtype=self.dtype)

    def __init__(self, dtype=np.float32):
        super(Accuracy, self).__init__(
            func=accuracy, batch_reduction=np.mean, whole_reduction=np.mean, dtype=dtype
        )


class CategoricalAccuracy(LambdaFunctionMetric):
    """
    Accuracy metric for one-hot encoded labels.
    """
    def lambda_function(self, predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
        return categorical_accuracy(predicted, truth, dtype=self.dtype)

    def __init__(self, dtype=np.float32):
        super(CategoricalAccuracy, self).__init__(
            func=categorical_accuracy, batch_reduction=np.mean, whole_reduction=np.mean, dtype=dtype
        )


class SparseCategoricalAccuracy(LambdaFunctionMetric):
    """
    Accuracy metric for integer labels.
    """
    def lambda_function(self, predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
        return sparse_categorical_accuracy(predicted, truth, dtype=self.dtype)

    def __init__(self, dtype=np.float32):
        super(SparseCategoricalAccuracy, self).__init__(
            func=sparse_categorical_accuracy, batch_reduction=np.mean, whole_reduction=np.mean, dtype=dtype,
        )


class BinaryAccuracy(LambdaFunctionMetric):

    def lambda_function(self, predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
        return binary_accuracy(predicted, truth, self.threshold, self.dtype)

    def __init__(self, dtype=np.float32, threshold=0.5):
        self.threshold = threshold
        super(BinaryAccuracy, self).__init__(
            func=binary_accuracy, batch_reduction=np.mean, whole_reduction=np.mean, dtype=dtype
        )


class MeanSquaredError(FunctionMetric):

    def __init__(self, const=0.5, dtype=np.float32):
        super(MeanSquaredError, self).__init__(
            func=SquaredError(const=const), batch_reduction=np.mean, whole_reduction=np.mean, dtype=dtype
        )
        self.name = 'MSE'


class MeanEuclideanError(LambdaFunctionMetric):

    def lambda_function(self, predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
        return mean_euclidean_error(predicted, truth, reduce=False, dtype=self.dtype)

    def __init__(self, dtype=np.float32):
        super(MeanEuclideanError, self).__init__(
            func=mean_euclidean_error, batch_reduction=np.mean, whole_reduction=np.mean, dtype=dtype,
        )
        self.name = 'MEE'


class RootMeanSquaredError(LambdaFunctionMetric):

    def lambda_function(self, predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
        return root_mean_squared_error(predicted, truth, dtype=self.dtype)

    def __init__(self, dtype=np.float32):
        super(RootMeanSquaredError, self).__init__(
            func=root_mean_squared_error, whole_reduction=np.mean, dtype=dtype,
        )
        self.name = 'RMSE'


# Utility aliases of mse, mee, rmse
MSE = MeanSquaredError
MEE = MeanEuclideanError
RMSE = RootMeanSquaredError


class Timing(FunctionMetric):
    """
    A metric that monitors elapsed time for each training step.
    """
    PRECISIONS = {'s', 'ns'}  # 's' -> perf_counter(), 'ns' -> perf_counter_ns()

    def get_time(self):
        return time.perf_counter() if self.precision == 's' else time.perf_counter_ns()

    def set_time(self):
        self.start_time = self.get_time()

    def time_function(self, predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
        return np.sum([self.get_time() - self.start_time], dtype=self.dtype)

    def __init__(self, precision=None, dtype=np.float32):
        """
        :param precision: If 's', a perf_counter() will be used for calculating time;
        otherwise, if 'ns', a pref_counter_ns() will be used.
        """
        precision = 's' if precision is None else precision
        if precision not in self.PRECISIONS:
            raise ValueError(f"Invalid precision value '{precision}': expected one of 's', 'ns'")
        self.precision = precision
        self.start_time = None
        super(Timing, self).__init__(func=self.time_function, whole_reduction=np.sum, dtype=dtype)

    def before_batch(self):
        self.set_time()


__all__ = [
    'Accuracy',
    'BinaryAccuracy',
    'CategoricalAccuracy',
    'SparseCategoricalAccuracy',
    'MeanSquaredError',
    'MeanEuclideanError',
    'RootMeanSquaredError',
    'MSE', 'MEE', 'RMSE',
    'Timing',
]
