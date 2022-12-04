# Regularized losses (L1, L2 etc.)
from __future__ import annotations
from ..utils import *


class Regularizer(Callable):
    """
    Base regularizator, applicable to a set of parameters.
    """
    @abstractmethod
    def update(self, x: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        """
        Base method used to get updates for a given (numpy array) parameter `x`.
        """
        pass

    @abstractmethod
    def loss(self, x: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        """
        Computes regularization loss term relative to parameter `x`.
        """
        pass

    def __call__(self, x: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        return self.loss(x, out)


class L1Regularizer(Regularizer):
    """
    L1 Regularizator, with the possibility to vary the subgradient used.
    """
    def __init__(self, subgradient_func: Callable[[np.ndarray], np.ndarray] = np.sign, l1_lambda: float = 0.01):
        """
        :param subgradient_func: Subgradient function to be used to supply
        a subgradient at each step. Defaults to np.sign.
        :param l1_lambda: Lambda constant for this regularization.
        """
        self.subgradient_func = subgradient_func
        self.l1_lambda = l1_lambda

    def __eq__(self, other):
        if not isinstance(other, L1Regularizer):
            return False
        return all([self.subgradient_func == other.subgradient_func, self.l1_lambda == other.l1_lambda])

    def update(self, x: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        result = self.l1_lambda * self.subgradient_func(x)
        if out is not None:
            np.copyto(dst=out, src=result)
            return out
        else:
            return result

    def loss(self, x: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        result = self.l1_lambda * np.sum(np.abs(x))
        if out is not None:
            np.copyto(dst=out, src=result)
            return out
        else:
            return result


class L2Regularizer(Regularizer):
    """
    L2 regularizator.
    """
    def __init__(self, l2_lambda: float = 0.01):
        self.l2_lambda = l2_lambda

    def __eq__(self, other):
        if not isinstance(other, L2Regularizer):
            return False
        return self.l2_lambda == other.l2_lambda

    def update(self, x: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        result = 2 * self.l2_lambda * x
        if out is not None:
            np.copyto(dst=out, src=result)
            return out
        else:
            return result

    def loss(self, x: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        result = self.l2_lambda * np.sum(np.square(x))
        if out is not None:
            np.copyto(dst=out, src=result)
            return out
        else:
            return result


class L1L2Regularizer(Regularizer):
    """
    Combined L1 and L2 regularizations.
    """
    def __init__(self, subgradient_func: Callable[[np.ndarray], np.ndarray] = np.sign, l1_lambda=0.01, l2_lambda=0.01):
        """
        :param subgradient_func: Subgradient function to be used to supply
        a subgradient at each step. Defaults to np.sign.
        :param l1_lambda: Lambda constant for L1 regularization.
        :param l2_lambda: Lambda constant for L2 regularization.
        """
        self.l1_regularizer = L1Regularizer(subgradient_func, l1_lambda)
        self.l2_regularizer = L2Regularizer(l2_lambda)

    def __eq__(self, other):
        if not isinstance(other, L1L2Regularizer):
            return False
        return all([self.l1_regularizer == other.l1_regularizer, self.l2_regularizer == other.l2_regularizer])

    def update(self, x: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        result = self.l1_regularizer.update(x) + self.l2_regularizer.update(x)
        if out is not None:
            np.copyto(dst=out, src=result)
            return out
        else:
            return result

    def loss(self, x: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        result = self.l1_regularizer.loss(x) + self.l2_regularizer.loss(x)
        if out is not None:
            np.copyto(dst=out, src=result)
            return out
        else:
            return result


__all__ = [
    'Regularizer',
    'L1Regularizer',
    'L2Regularizer',
    'L1L2Regularizer',
]
