# Commons metrics, wrapped around commons functions
from __future__ import annotations
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

    def get_name(self):
        return self.default_name()


class CategoricalAccuracy(FunctionMetric):
    """
    Accuracy metric for one-hot encoded labels.
    """
    def __init__(self, dtype=np.float64):
        super(CategoricalAccuracy, self).__init__(
            func=categorical_accuracy, batch_reduction=np.mean, whole_reduction=np.mean, dtype=dtype
        )
        self.func = lambda pred, truth: categorical_accuracy(pred, truth, dtype=dtype)

    def get_name(self):
        return self.default_name()


class BinaryAccuracy(FunctionMetric):

    def __init__(self, dtype=np.float64, threshold=0.5):
        super(BinaryAccuracy, self).__init__(
            func=binary_accuracy, batch_reduction=np.mean, whole_reduction=np.mean, dtype=dtype
        )
        self.func = lambda pred, truth: binary_accuracy(pred, truth, threshold, dtype)

    def get_name(self):
        return self.default_name()


class MeanSquaredError(FunctionMetric):

    def __init__(self, const=0.5, dtype=np.float64):
        super(MeanSquaredError, self).__init__(
            func=SquaredError(const=const), batch_reduction=np.mean, whole_reduction=np.mean, dtype=dtype
        )

    def get_name(self):
        return self.default_name()


class MeanEuclideanError(FunctionMetric):

    def __init__(self, dtype=np.float64):
        super(MeanEuclideanError, self).__init__(
            func=mean_euclidean_error, batch_reduction=np.mean, whole_reduction=np.mean, dtype=dtype,
        )
        self.func = lambda pred, truth: mean_euclidean_error(pred, truth, reduce=False, dtype=dtype)

    def get_name(self):
        return self.default_name()


class RootMeanSquaredError(FunctionMetric):

    def __init__(self, dtype=np.float64):
        super(RootMeanSquaredError, self).__init__(
            func=root_mean_squared_error, whole_reduction=np.mean, dtype=dtype,
        )
        self.func = lambda pred, truth: root_mean_squared_error(pred, truth, dtype=dtype)

    def get_name(self):
        return self.default_name()


# Utility aliases of mse, mee, rmse
MSE = MeanSquaredError
MEE = MeanEuclideanError
RMSE = RootMeanSquaredError


__all__ = [
    'Accuracy',
    'BinaryAccuracy',
    'CategoricalAccuracy',
    'MeanSquaredError',
    'MeanEuclideanError',
    'RootMeanSquaredError',
    'MSE', 'MEE', 'RMSE',
]
