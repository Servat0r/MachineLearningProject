# Metrics for monitoring a training/evaluation cycle
from __future__ import annotations
from core.utils.types import *


class Metric(Callable):
    """
    Base class for training/validation metrics.
    A Metric is a Callable that is used to calculate a function
    over the predicted and real data during training/validation.
    Metrics can also have an internal state, that is updated
    by the update() method, and may return result values both
    for an entire epoch or a train/eval/test batch.
    """

    def __init__(self, dtype=np.float32):
        self.name = self.default_name()
        self.dtype = dtype

    # Callbacks for metrics that need to maintain state over time
    def before_batch(self):
        pass

    def after_batch(self):
        pass

    def update(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        """
        Updates the internal state of the metric given a batch of data to evaluate.
        """
        pass

    def result(self, batch_num: int = None):
        pass

    def reset(self):
        pass

    def default_name(self):
        return str(type(self).__name__)

    def get_name(self):
        """
        The name that is used to identify the metric outside, e.g. in a History object.
        """
        return self.name

    def set_name(self, name: str):
        self.name = name

    def __call__(self, batch_num: int = None):
        return self.result(batch_num)


class FunctionMetric(Metric):
    """
    A Metric that wraps an existing function.
    Base class for MSE, MEE, MAE, RMSE.
    """

    def __init__(self, func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 batch_reduction: Callable[[np.ndarray], np.ndarray] = None,
                 # By default, batch results are not reduced
                 whole_reduction: Callable[[np.ndarray], np.ndarray] = np.mean,
                 # By default, we take the mean of all batch values
                 dtype=np.float32):
        """
        :param func: Function of the form (predictions, truth) -> result to be wrapped
        by this metric.
        :param batch_reduction: Function to apply to raw values given by the metric for each
        example in the current minibatch to return a unique value for the current minibatch.
        :param whole_reduction: The same as batch_reduction, but applied to values given
        by batch_reduction for each minibatch to return a unique value for the current epoch.
        :param dtype: Numpy datatype to be used for input and output values. Defaults to float64.
        """
        super(FunctionMetric, self).__init__(dtype=dtype)
        self.func = func
        self.batch_reduction = batch_reduction
        self.whole_reduction = whole_reduction
        self.values = []  # metric values for each minibatch

    def update(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        vals = self.func(pred, truth)
        vals = vals if self.batch_reduction is None else self.batch_reduction(vals).astype(self.dtype)
        self.values.append(vals)
        return vals

    def result(self, batch_num: int = None):
        if batch_num is not None:   # result over a batch
            return self.values[batch_num]
        else:   # result over all batches
            result = np.array(self.values, dtype=self.dtype)
            result = result if self.whole_reduction is None else self.whole_reduction(result)
            return result.astype(self.dtype)

    def reset(self):
        del self.values
        self.values = []


__all__ = [
    'Metric',
    'FunctionMetric',
]
