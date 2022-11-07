from __future__ import annotations

import numpy as np

from core.utils.types import *
from core.functions import *


class Loss:
    """
    Loss "Layer", representing the endpoint of the NN as a graph for computing gradients.
    """
    def __init__(self):
        self.input = None
        self.truth = None
        self.output = None

    def __call__(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        """
        Default call, returns raw values from forward pass.
        :param pred: Predicted values.
        :param truth: Ground truth values.
        :return:
        """
        return self.forward(pred, truth)

    def mean(self, pred: np.ndarray, truth: np.ndarray):
        """
        Calculate the average loss over a batch of predicted values and ground truth values.
        :param pred: Predicted values.
        :param truth: Ground truth values.
        :return:
        """
        sample_losses = self(pred, truth)
        return np.mean(sample_losses)

    def sum(self, pred: np.ndarray, truth: np.ndarray):
        sample_losses = self(pred, truth)
        return np.sum(sample_losses)

    @abstractmethod
    def forward(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self) -> np.ndarray:
        pass

    def clear(self):
        self.input = None
        self.truth = None
        self.output = None


# todo maybe it is better to rename it to NLLoss since it does NOT include Softmax
class CrossEntropyLoss(Loss):
    """
    Categorical Cross Entropy Loss. Note that this class does NOT include a Softmax layer.
    """
    def __init__(self, clip_value: TReal = 1e-7):
        super(CrossEntropyLoss, self).__init__()
        self.clip_value = clip_value

    def forward(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        self.input = pred
        self.truth = truth
        function = CategoricalCrossEntropy(truth, self.clip_value)
        return function(pred)

    def backward(self) -> np.ndarray:
        dvals = CategoricalCrossEntropy(self.truth, self.clip_value).grad()(self.input)
        dshape = dvals.shape
        return np.reshape(dvals, (dshape[0], dshape[2], dshape[1]))


class SoftmaxCrossEntropyLoss(Loss):
    """
    Softmax + CrossEntropy loss. It expects an input of raw, unnormalized values, and applies
    CrossEntropyLoss to the target distribution and normalized inputs through Softmax.
    """
    def __init__(self, const_shift=0, max_shift=False, clip_value=1e-7):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.softmax = Softmax(const_shift, max_shift)
        self.clip_value = clip_value

    def forward(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        self.input = pred
        cross_entropy = CategoricalCrossEntropy(truth, self.clip_value)
        smax_pred = self.softmax(pred)
        return cross_entropy(smax_pred)

    def backward(self) -> np.ndarray:
        pass


class MSELoss(Loss):
    """
    Mean Square Error Loss.
    """
    def forward(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        self.input = pred
        return SquareError()(pred - truth)  # todo sure?

    def backward(self) -> np.ndarray:
        dvals = SquareError().grad()(self.input)
        dshape = dvals.shape
        dvals = np.reshape(dvals, (dshape[0], dshape[2], dshape[1]))
        return dvals


class MeanAbsErrorLoss(Loss):
    """
    Mean Absolute error Loss.
    """
    def forward(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        self.input = pred
        return AbsError()(pred - truth)  # todo sure?

    def backward(self) -> np.ndarray:
        pass


__all__ = [
    'Loss',
    'CrossEntropyLoss',
    'SoftmaxCrossEntropyLoss',
    'MSELoss',
    'MeanAbsErrorLoss',
]