from __future__ import annotations
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


class CrossEntropyLoss(Loss):
    """
    Categorical Cross Entropy Loss. Note that this class does NOT include a Softmax layer.
    """
    def __init__(self, clip_value: TReal = 1e-7):
        super(CrossEntropyLoss, self).__init__()
        self.clip_value = clip_value
        self.func = None

    def forward(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        self.input = pred
        self.truth = truth
        self.func = CategoricalCrossEntropy(self.truth, self.clip_value)
        return self.func(pred)

    def backward(self) -> np.ndarray:
        return self.func.grad(self.func)(self.input)


class NLLoss(Loss):
    """
    Negative Log Likelihoods Loss, assuming that the target distribution
    is of the form [..., 1, ...] (e.g. one-hot encoded labels).
    """
    def __init__(self, clip_value=1e-7):
        super(NLLoss, self).__init__()
        self.clip_value = clip_value
        self.func = None

    def forward(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        """
        It is assumed here that truth has shape 1.
        """
        trshape = truth.shape
        ltrs = len(trshape)
        # Convert one-hot encoded labels to "regular" ones
        norm_truth = np.argmax(truth, axis=ltrs-1) if ltrs >= 2 else truth
        self.input = pred
        self.func = CategoricalCrossEntropy(norm_truth, self.clip_value)
        return self.func(pred)

    def backward(self) -> np.ndarray:
        return self.func.grad(self.func)(self.input)


class SoftmaxCrossEntropyLoss(Loss):
    """
    Softmax + CrossEntropy loss. It expects an input of raw, unnormalized values, and applies
    CrossEntropyLoss to the target distribution and normalized inputs through Softmax.
    """
    def __init__(self, const_shift=0, max_shift=False, clip_value=1e-7):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.softmax = Softmax(const_shift, max_shift)
        self.net = None
        self.clip_value = clip_value
        self.func = None

    def forward(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        self.input = pred
        if self.func is None:
            self.func = CategoricalCrossEntropy(truth, self.clip_value)
        else:
            self.func.set_truth_values(truth)
        self.net = self.softmax(pred)
        return self.func(self.net)

    def backward(self) -> np.ndarray:
        dvals: np.ndarray = self.func.grad()(self.net)
        # noinspection PyArgumentList
        return self.softmax.grad()(self.input, dvals)


class SquaredErrorLoss(Loss):
    """
    Squared Error Loss. Default .forward(...) method computes the squared error
    for each pattern SEPARATELY with NO reduction, while .mean(...) and .sum(...)
    compute respectively the average error over training examples and the sum
    over them.
    """
    func = SquareError()

    def forward(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        self.input = pred
        return self.func(pred - truth)  # todo sure?

    def backward(self) -> np.ndarray:
        return self.func.grad()(self.input)


class MSELoss(Loss):
    """
    Mean Squared Error Loss over a batch of training examples. Its .forward(...)
    method is equivalent to SquaredErrorLoss.mean(...).
    """
    func = SquareError()

    def forward(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        self.input = pred
        return np.mean(self.func(pred - truth))  # todo sure?

    def backward(self) -> np.ndarray:
        return self.func.grad()(self.input) / len(self.input)


class MeanAbsErrorLoss(Loss):
    """
    Mean Absolute error Loss.
    """
    func = AbsError()

    def forward(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        self.input = pred
        return self.func(pred - truth)  # todo sure?

    def backward(self) -> np.ndarray:
        pass


__all__ = [
    'Loss',
    'CrossEntropyLoss',
    'NLLoss',
    'SoftmaxCrossEntropyLoss',
    'SquaredErrorLoss',
    'MSELoss',
    'MeanAbsErrorLoss',
]
