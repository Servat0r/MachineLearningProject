from __future__ import annotations
from core.utils.types import *
import core.functions as cf
from .layers import Dense, Layer, Linear


class Loss:
    """
    Loss "Layer", representing the endpoint of the NN as a graph for computing gradients.
    """
    REDUCTIONS = {'none', 'mean', 'sum'}

    def __init__(self, reduction='mean', dtype=np.float64):
        self.dtype = dtype
        if reduction is not None and reduction not in self.REDUCTIONS:
            raise ValueError(f"Unknown reduction type '{reduction}'")
        else:
            self.reduction = reduction

    def __call__(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        """
        Default call, returns raw values from forward pass.
        :param pred: Predicted values.
        :param truth: Ground truth values.
        :return:
        """
        return self.forward(pred, truth)

    @abstractmethod
    def forward(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, dvals: np.ndarray, truth: np.ndarray) -> np.ndarray:
        pass

    def reduce(self, y: np.ndarray):
        """
        Applies given reduction to the raw loss output.
        """
        if self.reduction == 'mean':
            return np.mean(y, axis=0, dtype=self.dtype)
        elif self.reduction == 'sum':
            return np.sum(y, axis=0, dtype=self.dtype)
        else:
            return y.astype(self.dtype)


class CrossEntropyLoss(Loss):   # todo need to check with a classification problem
    """
    Categorical Cross Entropy Loss. Note that this class does NOT include a Softmax layer.
    """
    def __init__(self, clip_value: TReal = 1e-7, reduction='mean', dtype=np.float64):
        super(CrossEntropyLoss, self).__init__(reduction=reduction, dtype=dtype)
        self.clip_value = clip_value
        self.func = cf.CategoricalCrossEntropy(self.clip_value)

    def forward(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        y = self.func(pred, truth).astype(self.dtype)
        return self.reduce(y)

    def backward(self, dvals: np.ndarray, truth: np.ndarray) -> np.ndarray:
        return self.func.grad(dvals, truth).astype(self.dtype)


class MSELoss(Loss):
    """
    Mean Squared Error Loss over a batch of training examples. Its .forward(...)
    method is equivalent to SquaredErrorLoss.mean(...).
    """

    def __init__(self, const=0.5, reduction='mean', dtype=np.float64):
        super(MSELoss, self).__init__(reduction=reduction, dtype=dtype)
        self.const = const

    def forward(self, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
        y = self.const * np.sum((truth - pred)**2, axis=-1)
        return self.reduce(y)

    def backward(self, dvals: np.ndarray, truth: np.ndarray) -> np.ndarray:
        result = -2 * self.const * (truth - dvals)
        return result.astype(result)


class RegularizedLoss(Loss):
    """
    A loss with a regularization term.
    """
    def __init__(self, base_loss: Loss):
        super(RegularizedLoss, self).__init__(dtype=base_loss.dtype)
        self.base_loss = base_loss

    def __call__(self, pred: np.ndarray, truth: np.ndarray,
                 layers: Iterable[Layer] = None) -> tuple[np.ndarray, np.ndarray]:
        return self.forward(pred, truth, layers)

    def regularization_loss(self, layers: Layer | Iterable[Layer]) -> np.ndarray:
        if isinstance(layers, Dense):
            return self.regularization_loss(layers.linear)
        elif isinstance(layers, Layer):
            result = np.zeros(1)
            if isinstance(layers, Linear) and layers.is_trainable():
                if layers.weights_regularizer is not None:
                    result += layers.weights_regularizer.loss(layers.weights)
                if layers.biases_regularizer is not None:
                    result += layers.biases_regularizer.loss(layers.biases)
            return result.astype(self.dtype)
        elif isinstance(layers, Iterable):
            result = np.zeros(1)
            for layer in layers:
                result += self.regularization_loss(layer)
            return result.astype(self.dtype)
        else:
            raise TypeError(f"Invalid type {type(layers)}: allowed ones are {Layer} or {Iterable[Layer]}")

    def forward(self, pred: np.ndarray, truth: np.ndarray,
                layers: Iterable[Layer] = None) -> tuple[np.ndarray, np.ndarray]:
        data_loss = self.base_loss.forward(pred, truth)
        reg_loss = self.regularization_loss(layers)
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
