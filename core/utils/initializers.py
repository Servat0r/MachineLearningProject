# Initializers for NN weights
from __future__ import annotations
from .types import *


# TypeVar for initialization policies
TInit = TypeVar('TInit', bound=Callable[[tuple, Optional[tuple], ...], tuple[np.ndarray, np.ndarray, ...]])


class Initializer:

    def __call__(self, weights_shape: TShape, biases_shape: TShape = None, dtype=np.float64, *args, **kwargs):
        return self.initialize(weights_shape, biases_shape, dtype, *args, **kwargs)

    @abstractmethod
    def initialize(self, weights_shape: TShape, biases_shape: TShape = None, dtype=np.float64,
                   *args, **kwargs) -> tuple[np.ndarray, Optional[np.ndarray]]:
        pass


class ZeroInitializer(Initializer):
    """
    An initializer that sets all weights and biases to 0.
    """
    def initialize(self, weights_shape: TShape, biases_shape: TShape = None, dtype=np.float64,
                   *args, **kwargs) -> tuple[np.ndarray, Optional[np.ndarray]]:
        weights = np.zeros(weights_shape, dtype=dtype)
        biases = np.zeros(biases_shape, dtype=dtype) if biases_shape is not None else None
        return weights, biases


class RandomUniformInitializer(Initializer):
    """
    Random uniform initializer within a specified range [p, q].
    """
    def __init__(self, low: TReal = 0.0, high: TReal = 1.0, zero_bias=False):
        self.low = low
        self.high = high
        self.zero_bias = zero_bias

    def initialize(self, weights_shape: TShape, biases_shape: TShape = None, dtype=np.float64,
                   *args, **kwargs) -> tuple[np.ndarray, Optional[np.ndarray]]:
        weights = np.random.uniform(self.low, self.high, weights_shape).astype(dtype=dtype)
        if biases_shape is not None:
            if self.zero_bias:
                biases = np.zeros(biases_shape, dtype=dtype)
            else:
                biases = np.random.uniform(self.low, self.high, biases_shape).astype(dtype=dtype)
        else:
            biases = None
        return weights, biases


class RandomNormalDefaultInitializer(Initializer):
    """
    Random normal (Gaussian) initializer (mean = 0, std = 1).
    """
    def __init__(self, scale: TReal = 1.0, zero_bias=False):
        self.scale = scale
        self.zero_bias = zero_bias

    def initialize(self, weights_shape: TShape, biases_shape: TShape = None, dtype=np.float64,
                   *args, **kwargs) -> tuple[np.ndarray, Optional[np.ndarray]]:
        weights = self.scale * np.random.randn(*weights_shape)
        if biases_shape is not None:
            if self.zero_bias:
                biases = np.zeros(biases_shape, dtype=dtype)
            else:
                biases = self.scale * np.random.randn(*biases_shape)
        else:
            biases = None
        return weights, biases


# TODO Add RandomNormal with mean and std, Xavier


__all__ = [
    'Initializer',
    'ZeroInitializer',
    'RandomUniformInitializer',
    'RandomNormalDefaultInitializer',
]