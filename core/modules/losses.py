from __future__ import annotations
from core.utils.types import *
import core.functions as cf
from .layers import Dense, Layer, Linear


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
    def backward(self, dvals: np.ndarray, truth: np.ndarray) -> np.ndarray:
        pass

    def clear(self):
        self.input = None
        self.truth = None
        self.output = None


class CrossEntropyLoss(Loss):   # todo need to check with a classification problem
    """
    Categorical Cross Entropy Loss. Note that this class does NOT include a Softmax layer.
    """
    def __init__(self, clip_value: TReal = 1e-7):
        super(CrossEntropyLoss, self).__init__()
        self.clip_value = clip_value
        self.func = cf.CategoricalCrossEntropy(self.clip_value)

    def forward(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        self.input = pred
        self.truth = truth
        return self.func(self.input, self.truth)

    def backward(self, dvals: np.ndarray, truth: np.ndarray) -> np.ndarray:
        self.truth = truth
        return self.func.grad(dvals, self.truth)


class MSELoss(Loss):
    """
    Mean Squared Error Loss over a batch of training examples. Its .forward(...)
    method is equivalent to SquaredErrorLoss.mean(...).
    """
    REDUCTIONS = {'none', 'mean', 'sum'}

    def __init__(self, const=0.5, reduction='mean'):
        super(MSELoss, self).__init__()
        self.const = const
        if reduction is not None and reduction not in self.REDUCTIONS:
            raise ValueError(f"Unknown reduction type '{reduction}'")
        else:
            self.reduction = reduction

    def forward(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        y = self.const * np.sum((truth - pred)**2, axis=-1)
        if self.reduction == 'sum':
            return np.sum(y, axis=0)
        elif self.reduction == 'mean':
            return np.mean(y, axis=0)
        else:
            return y

    def backward(self, dvals: np.ndarray, truth: np.ndarray) -> np.ndarray:
        return -2 * self.const * (truth - dvals)


class RegularizedLoss(Loss):
    """
    A loss with a regularization term.
    """
    def __init__(self, base_loss: Loss, layers: Iterable[Layer]):
        super(RegularizedLoss, self).__init__()
        self.base_loss = base_loss
        self.layers = layers

    def regularization_loss(self, layers: Layer | Iterable[Layer]) -> np.ndarray:
        if isinstance(layers, Dense):
            return self.regularization_loss(layers.linear)
        elif isinstance(layers, Layer):
            result = np.zeros(1)
            if isinstance(layers, Linear) and layers.is_parametrized():
                if layers.weights_regularizer is not None:
                    result += layers.weights_regularizer.loss(layers.weights)
                if layers.biases_regularizer is not None:
                    result += layers.biases_regularizer.loss(layers.biases)
            return result
        elif isinstance(layers, Iterable):
            result = np.zeros(1)
            for layer in layers:
                result += self.regularization_loss(layer)
            return result
        else:
            raise TypeError(f"Invalid type {type(layers)}: allowed ones are {Layer} or {Iterable[Layer]}")

    def forward(self, pred: np.ndarray, truth: np.ndarray,
                target_shape: tuple = (1,)) -> tuple[np.ndarray, np.ndarray]:
        data_loss = self.base_loss.forward(pred, truth)
        reg_loss = self.regularization_loss(self.layers)
        return data_loss, reg_loss

    def backward(self, dvals: np.ndarray, truth: np.ndarray) -> np.ndarray:
        # Backward pass "direct" handling (i.e., by updating gradients of the weights)
        # without an underlying computational graph is complicated
        # For regularizers like L1, the actual backward pass happens when calling
        # backward() on the previous layers.
        return self.base_loss.backward(dvals, truth)


__all__ = [
    'Loss',
    'CrossEntropyLoss',
    'MSELoss',
    'RegularizedLoss',
]
