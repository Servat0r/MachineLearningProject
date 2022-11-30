# Commons metrics, wrapped around commons functions
from __future__ import annotations
import time
from core.utils.types import *
from core.functions import *
from .base import *


class Accuracy(FunctionMetric):
    """
    Accuracy metric for numerical (scalar) labels.
    """
    def __init__(self, dtype=np.float64):
        super(Accuracy, self).__init__(
            func=accuracy, batch_reduction=np.mean, whole_reduction=np.mean, dtype=dtype
        )
        self.func = lambda pred, truth: accuracy(pred, truth, dtype=dtype)


class CategoricalAccuracy(FunctionMetric):
    """
    Accuracy metric for one-hot encoded labels.
    """
    def __init__(self, dtype=np.float64):
        super(CategoricalAccuracy, self).__init__(
            func=categorical_accuracy, batch_reduction=np.mean, whole_reduction=np.mean, dtype=dtype
        )
        self.func = lambda pred, truth: categorical_accuracy(pred, truth, dtype=dtype)


class BinaryAccuracy(FunctionMetric):

    def __init__(self, dtype=np.float64, threshold=0.5):
        super(BinaryAccuracy, self).__init__(
            func=binary_accuracy, batch_reduction=np.mean, whole_reduction=np.mean, dtype=dtype
        )
        self.func = lambda pred, truth: binary_accuracy(pred, truth, threshold, dtype)


class MeanSquaredError(FunctionMetric):

    def __init__(self, const=0.5, dtype=np.float64):
        super(MeanSquaredError, self).__init__(
            func=SquaredError(const=const), batch_reduction=np.mean, whole_reduction=np.mean, dtype=dtype
        )
        self.name = 'MSE'


class MeanEuclideanError(FunctionMetric):

    def __init__(self, dtype=np.float64):
        super(MeanEuclideanError, self).__init__(
            func=mean_euclidean_error, batch_reduction=np.mean, whole_reduction=np.mean, dtype=dtype,
        )
        self.func = lambda pred, truth: mean_euclidean_error(pred, truth, reduce=False, dtype=dtype)
        self.name = 'MEE'


class RootMeanSquaredError(FunctionMetric):

    def __init__(self, dtype=np.float64):
        super(RootMeanSquaredError, self).__init__(
            func=root_mean_squared_error, whole_reduction=np.mean, dtype=dtype,
        )
        self.func = lambda pred, truth: root_mean_squared_error(pred, truth, dtype=dtype)
        self.name = 'RMSE'


# Utility aliases of mse, mee, rmse
MSE = MeanSquaredError
MEE = MeanEuclideanError
RMSE = RootMeanSquaredError


class Timing(FunctionMetric):

    PRECISIONS = {'s', 'ns'}  # 's' -> perf_counter(), 'ns' -> perf_counter_ns()

    def get_time(self):
        return time.perf_counter() if self.precision == 's' else time.perf_counter_ns()

    def set_time(self):
        self.start_time = self.get_time()

    def time_fun(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        return np.sum([self.get_time() - self.start_time], dtype=self.dtype)
        # return np.array(self.get_time() - self.start_time, dtype=self.dtype)

    def __init__(self, precision=None, dtype=np.float64):
        precision = 's' if precision is None else precision
        if precision not in self.PRECISIONS:
            raise ValueError(f"Invalid precision value '{precision}': expected one of 's', 'ns'")
        self.precision = precision
        self.start_time = None
        super(Timing, self).__init__(func=self.time_fun, whole_reduction=np.sum, dtype=dtype)

    def before_batch(self):
        self.set_time()


__all__ = [
    'Accuracy',
    'BinaryAccuracy',
    'CategoricalAccuracy',
    'MeanSquaredError',
    'MeanEuclideanError',
    'RootMeanSquaredError',
    'MSE', 'MEE', 'RMSE',
    'Timing',
]
