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
        """
        :param reduction: Reduction to apply to calculated values for each example.
        If 'mean', applies np.mean(); if 'sum', applies np.sum(); otherwise, no
        reduction is applied.
        """
        self.dtype = dtype
        if reduction is not None and reduction not in self.REDUCTIONS:
            raise ValueError(f"Unknown reduction type '{reduction}'")
        else:
            self.reduction = reduction

    def __eq__(self, other):
        if not isinstance(other, Loss):
            return False
        return all([self.dtype == other.dtype, self.reduction == other.reduction])

    def __call__(self, predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
        """
        Default call, returns raw values from forward pass.
        :param predicted: Predicted values.
        :param truth: Ground truth values.
        :return:
        """
        return self.forward(predicted, truth)

    @abstractmethod
    def forward(self, predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, delta_vals: np.ndarray, truth: np.ndarray) -> np.ndarray:
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


class CrossEntropyLoss(Loss):
    """
    Categorical Cross Entropy Loss. Note that this class does NOT include a Softmax layer.
    """
    def __init__(self, clip_value: TReal = 1e-7, reduction='mean', dtype=np.float64):
        super(CrossEntropyLoss, self).__init__(reduction=reduction, dtype=dtype)
        self.clip_value = clip_value
        self.func = cf.CategoricalCrossEntropy(self.clip_value)

    def __eq__(self, other):
        if not isinstance(other, CrossEntropyLoss):
            return False
        return all([
            super(CrossEntropyLoss, self).__eq__(other), self.clip_value == other.clip_value,
            self.func == other.func,
        ])

    def forward(self, predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
        y = self.func(predicted, truth).astype(self.dtype)
        return self.reduce(y)

    def backward(self, delta_vals: np.ndarray, truth: np.ndarray) -> np.ndarray:
        return self.func.grad(delta_vals, truth).astype(self.dtype)


class MSELoss(Loss):
    """
    Mean Squared Error Loss over a batch of training examples.
    """

    def __init__(self, const=0.5, reduction='mean', dtype=np.float64):
        super(MSELoss, self).__init__(reduction=reduction, dtype=dtype)
        self.func = cf.SquaredError(const=const)

    def __eq__(self, other):
        if not isinstance(other, MSELoss):
            return False
        return all([super(MSELoss, self).__eq__(other), self.func == other.func])

    def forward(self, predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
        return self.reduce(self.func(predicted, truth))

    def backward(self, delta_vals: np.ndarray, truth: np.ndarray) -> np.ndarray:
        return self.func.grad(delta_vals, truth).astype(self.dtype)


class RegularizedLoss(Loss):
    """
    A loss with a regularization term. This loss behaves by wrapping a "base" one
    and returning a couple (data_loss, regularization_loss) each time it is called
    with __call__() of forward(). Regularization term is calculated by scanning a
    layer or a sequence of layers given as third parameter to the previous methods
    and summing up updates given by all regularizers associated with them.
    Backpropagation of regularization terms is left to layers themselves for simplicity.
    """
    def __init__(self, base_loss: Loss):
        super(RegularizedLoss, self).__init__(dtype=base_loss.dtype)
        self.base_loss = base_loss

    def __eq__(self, other):
        if not isinstance(other, RegularizedLoss):
            return False
        return all([super(RegularizedLoss, self).__eq__(other), self.base_loss == other.base_loss])

    def __call__(self, predicted: np.ndarray, truth: np.ndarray,
                 layers: Iterable[Layer] = None) -> tuple[np.ndarray, np.ndarray]:
        return self.forward(predicted, truth, layers)

    def regularization_loss(self, layers: Layer | Iterable[Layer]) -> np.ndarray:
        """
        Computes regularization term given a set of layers as described
        in class documentation.
        """
        if isinstance(layers, Dense):
            # Only linear layer inside a Dense has regularization term
            return self.regularization_loss(layers.linear)
        elif isinstance(layers, Layer):
            result = np.zeros(1)
            if isinstance(layers, Linear) and layers.is_trainable():
                # Update result accordingly to the specific regularizers of this layer
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

    def forward(self, predicted: np.ndarray, truth: np.ndarray,
                layers: Iterable[Layer] = None) -> tuple[np.ndarray, np.ndarray]:
        data_loss = self.base_loss.forward(predicted, truth)
        regularization_loss = self.regularization_loss(layers)
        return data_loss, regularization_loss

    def backward(self, delta_vals: np.ndarray, truth: np.ndarray) -> np.ndarray:
        # Backward pass "direct" handling (i.e., by updating gradients of the weights)
        # without an underlying computational graph is complicated
        # For regularizers like L1, the actual backward pass happens when calling
        # backward() on the previous layers.
        return self.base_loss.backward(delta_vals, truth)


__all__ = [
    'Loss',
    'CrossEntropyLoss',
    'MSELoss',
    'RegularizedLoss',
]
