# Metrics for monitoring a training/evaluation cycle
from __future__ import annotations
from core.utils.types import *


class Metric(Callable):

    def __init__(self, dtype=np.float64):
        self.dtype = dtype

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
        return self.default_name()

    def __call__(self, batch_num: int = None):
        return self.result(batch_num)


class FunctionMetric(Metric):

    def __init__(self, func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 batch_reduction: Callable[[np.ndarray], np.ndarray] = lambda x: x,
                 # By default, batch results are not reduced
                 whole_reduction: Callable[[np.ndarray], np.ndarray] = np.mean,
                 # By default, we take the mean of all batch values
                 dtype=np.float64):
        super(FunctionMetric, self).__init__(dtype=dtype)
        self.func = func
        self.batch_reduction = batch_reduction
        self.whole_reduction = whole_reduction
        self.values = []
        # list of array losses (todo we can convert to array by using np.c_ every time or knowing dataset size)

    def update(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        vals = self.func(pred, truth).astype(self.dtype)
        self.values.append(vals)
        return self.batch_reduction(vals).astype(self.dtype)

    def result(self, batch_num: int = None):
        if batch_num is not None:   # result over a batch
            vals = self.values[batch_num]
            return self.batch_reduction(vals).astype(self.dtype)
        else:   # result over all batches
            all_vals = np.column_stack(self.values)
            return self.whole_reduction(all_vals).astype(self.dtype)

    def get_name(self):
        return f"{type(self).__name__}<{self.func}>"

    def reset(self):
        del self.values
        self.values = []


__all__ = [
    'Metric',
    'FunctionMetric',
]
