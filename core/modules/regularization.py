# Regularized losses (L1, L2 etc.)
from __future__ import annotations
from ..utils import *


class Regularizer(Callable):
    """
    Base regularizator, applicable to a set of parameters.
    """
    def __init__(self):
        pass

    @abstractmethod
    def update(self, x: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        pass

    @abstractmethod
    def loss(self, x: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        pass

    def __call__(self, x: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        return self.loss(x, out)


class L1Regularizer(Regularizer):
    """
    L1 Regularizator, with the possibility to vary the subgradient used.
    """
    def __init__(self, subgrad_func: Callable[[np.ndarray], np.ndarray] = np.sign, l1_lambda: float = 0.01):
        super(L1Regularizer, self).__init__()
        self.subgrad_func = subgrad_func
        self.l1_lambda = l1_lambda

    def update(self, x: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        result = self.l1_lambda * self.subgrad_func(x)
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
        super(L2Regularizer, self).__init__()
        self.l2_lambda = l2_lambda

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

    def __init__(self, subgrad_func: Callable[[np.ndarray], np.ndarray] = np.sign, l1_lambda=0.01, l2_lambda=0.01):
        super(L1L2Regularizer, self).__init__()
        self.l1_reg = L1Regularizer(subgrad_func, l1_lambda)
        self.l2_reg = L2Regularizer(l2_lambda)

    def update(self, x: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        result = self.l1_reg.update(x) + self.l2_reg.update(x)
        if out is not None:
            np.copyto(dst=out, src=result)
            return out
        else:
            return result

    def loss(self, x: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        result = self.l1_reg.loss(x) + self.l2_reg.loss(x)
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
