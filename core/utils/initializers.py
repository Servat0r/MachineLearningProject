# Initializers for NN weights
from __future__ import annotations
import math
from .types import *


# TypeVar for initialization policies
TInit = TypeVar('TInit', bound=Callable[[tuple, Optional[tuple], ...], tuple[np.ndarray, np.ndarray, ...]])


class Initializer:

    @abstractmethod
    def __call__(self, shape: TShape, dtype=np.float32, *args, **kwargs) -> np.ndarray:
        pass


class ZeroInitializer(Initializer):
    def __call__(self, shape: TShape, dtype=np.float32, *args, **kwargs) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)


class RandomUniformInitializer(Initializer):
    """
    Random uniform initializer within a specified range [p, q].
    """
    def __init__(self, low: TReal = -0.1, high: TReal = 0.1, seed=None):
        self.low = low
        self.high = high
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self, shape: TShape, dtype=np.float32, *args, **kwargs) -> np.ndarray:
        return self.rng.uniform(self.low, self.high, shape).astype(dtype=dtype)


class RandomNormalDefaultInitializer(Initializer):
    """
    Random normal (Gaussian) initializer (mean = 0, std = 1).
    """
    def __init__(self, scale: TReal = 1.0, seed=None):
        self.scale = scale
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self, shape: TShape, dtype=np.float32, *args, **kwargs) -> np.ndarray:
        result = self.scale * self.rng.normal(loc=0.0, scale=self.scale, size=shape)
        return result.astype(dtype=dtype)


class FanInitializer(Initializer):
    """
    Initializer that uses fan-in/out values to initialize weights.
    """
    def __init__(self, fan_value: int, seed=None):
        self.fan_value = 1 / math.sqrt(fan_value)
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self, shape: TShape, dtype=np.float32, fan: int = 8, *args, **kwargs):
        return self.rng.uniform(-self.fan_value, self.fan_value, shape).astype(dtype=dtype)


__all__ = [
    'Initializer',
    'ZeroInitializer',
    'RandomUniformInitializer',
    'RandomNormalDefaultInitializer',
    'FanInitializer',
]
