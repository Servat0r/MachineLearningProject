# Metrics for monitoring a training/evaluation cycle
from __future__ import annotations
from core.utils.types import *


class Metric(Callable):

    def __init__(self, dtype=np.float64):
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
        return self.name

    def set_name(self, name: str):
        self.name = name

    def __call__(self, batch_num: int = None):
        return self.result(batch_num)


class FunctionMetric(Metric):

    def __init__(self, func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 batch_reduction: Callable[[np.ndarray], np.ndarray] = None,
                 # By default, batch results are not reduced
                 whole_reduction: Callable[[np.ndarray], np.ndarray] = np.mean,
                 # By default, we take the mean of all batch values
                 dtype=np.float64):
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
