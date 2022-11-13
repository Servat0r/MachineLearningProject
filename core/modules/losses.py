from __future__ import annotations
from core.utils.types import *
import core.diffs as dfs
import core.functions as cf
from .regularization import *


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
        self.func = cf.CategoricalCrossEntropy(self.truth, self.clip_value)
        return self.func(self.input)

    def backward(self) -> np.ndarray:
        return dfs.grad(type(self.func), self.func, self.input)


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
        self.truth = np.argmax(truth, axis=ltrs-1) if ltrs >= 2 else truth
        self.input = pred
        self.func = cf.CategoricalCrossEntropy(self.truth, self.clip_value)
        return self.func(self.input)

    def backward(self) -> np.ndarray:
        return dfs.grad(type(self.func), self.func, self.input, self.truth, self.clip_value)


class SoftmaxCrossEntropyLoss(Loss):
    """
    Softmax + CrossEntropy loss. It expects an input of raw, unnormalized values, and applies
    CrossEntropyLoss to the target distribution and normalized inputs through Softmax.
    """
    def __init__(self, const_shift=0, max_shift=False, clip_value=1e-7):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.const_shift = const_shift
        self.max_shift = max_shift
        self.clip_value = clip_value
        self.net = None
        self.softmax = cf.Softmax(const_shift, max_shift)
        self.cross_entropy = None

    def forward(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        self.input = pred
        self.net = self.softmax(pred)
        self.truth = truth
        self.cross_entropy = cf.CategoricalCrossEntropy(self.truth, self.clip_value)
        return self.cross_entropy(self.net)

    def backward(self) -> np.ndarray:
        dvals: np.ndarray = dfs.grad(type(self.cross_entropy), self.cross_entropy, self.net)
        # noinspection PyArgumentList
        return dfs.vjp(type(self.softmax), self.softmax, self.input, dvals)


class SquaredErrorLoss(Loss):
    """
    Squared Error Loss. Default .forward(...) method computes the squared error
    for each pattern SEPARATELY with NO reduction, while .mean(...) and .sum(...)
    compute respectively the average error over training examples and the sum
    over them.
    """
    def __init__(self, const=0.5):
        super(SquaredErrorLoss, self).__init__()
        self.const = const
        self.func = None

    def forward(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        self.input = pred
        self.truth = truth
        self.func = cf.SquareError(self.truth, self.const)
        return self.func(self.input)

    def backward(self) -> np.ndarray:
        return dfs.grad(type(self.func), self.func, self.input)


class MSELoss(Loss):
    """
    Mean Squared Error Loss over a batch of training examples. Its .forward(...)
    method is equivalent to SquaredErrorLoss.mean(...).
    """
    def __init__(self, const=0.5):
        super(MSELoss, self).__init__()
        self.const = const
        self.func = None

    def forward(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        self.input = pred
        self.truth = truth
        self.func = cf.MeanSquareError(self.truth, self.const)
        return self.func(self.input)

    def backward(self) -> np.ndarray:
        return dfs.grad(type(self.func), self.func, self.input)


class RegularizedLoss(Loss):
    """
    A loss with a regularization term.
    """
    def __init__(self, base_loss: Loss, regularizers: Regularizer | Iterable[Regularizer]):
        super(RegularizedLoss, self).__init__()
        self.base_loss = base_loss
        regularizers = {regularizers} if isinstance(regularizers, Regularizer) else regularizers
        self.regularizers = regularizers

    def forward(self, pred: np.ndarray, truth: np.ndarray,
                target_shape: tuple = (1,)) -> np.ndarray:
        loss_fwd = self.base_loss.forward(pred, truth)
        reg_fwd = np.zeros(target_shape)
        for regularizer in self.regularizers:
            regularizer(target_shape, reg_fwd)
        if isinstance(target_shape, int) or all([len(target_shape) == 1, target_shape[0] == 1]):
            reg_fwd = reg_fwd.item()
        loss_fwd += reg_fwd
        return loss_fwd

    def backward(self) -> np.ndarray:
        # Backward pass "direct" handling (i.e., by updating gradients of the weights)
        # without an underlying computational graph is complicated
        # For regularizers like L1, the actual backward pass happens when calling
        # update_param_grads() for each regularizer.
        return self.base_loss.backward()


__all__ = [
    'Loss',
    'CrossEntropyLoss',
    'NLLoss',
    'SoftmaxCrossEntropyLoss',
    'SquaredErrorLoss',
    'MSELoss',
    'RegularizedLoss',
]
