# Initializers for NN weights
from __future__ import annotations
from .types import *


# TypeVar for initialization policies
TInit = TypeVar('TInit', bound=Callable[[tuple, Optional[tuple], ...], tuple[np.ndarray, np.ndarray, ...]])


class Initializer:

    @abstractmethod
    def __call__(self, shape: TShape, dtype=np.float64, *args, **kwargs) -> np.ndarray:
        pass


class ZeroInitializer(Initializer):
    def __call__(self, shape: TShape, dtype=np.float64, *args, **kwargs) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)


class RandomUniformInitializer(Initializer):
    """
    Random uniform initializer within a specified range [p, q].
    """
    def __init__(self, low: TReal = 0.0, high: TReal = 1.0):
        self.low = low
        self.high = high

    def __call__(self, shape: TShape, dtype=np.float64, *args, **kwargs) -> np.ndarray:
        return np.random.uniform(self.low, self.high, shape).astype(dtype=dtype)


class RandomNormalDefaultInitializer(Initializer):
    """
    Random normal (Gaussian) initializer (mean = 0, std = 1).
    """
    def __init__(self, scale: TReal = 1.0):
        self.scale = scale

    def __call__(self, shape: TShape, dtype=np.float64, *args, **kwargs) -> np.ndarray:
        result = self.scale * np.random.randn(*shape)
        return result.astype(dtype=dtype)


__all__ = [
    'Initializer',
    'ZeroInitializer',
    'RandomUniformInitializer',
    'RandomNormalDefaultInitializer',
]
