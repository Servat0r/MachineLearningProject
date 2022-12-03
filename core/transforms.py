from __future__ import annotations
from core.utils.types import *


class Transform(Callable):

    @abstractmethod
    def transform(self, x: np.ndarray):
        pass

    @abstractmethod
    def inverse_transform(self, x: np.ndarray):
        pass

    def __call__(self, x: np.ndarray):
        return self.transform(x)


class OneHotEncoder(Transform):

    def transform(self, x: np.ndarray):
        n_labels = np.max(x) + 1
        n_samples = len(x)
        out_shape = x.shape + (n_labels,)
        out = np.zeros(out_shape)
        out[range(n_samples), x] = 1
        return out

    def inverse_transform(self, x: np.ndarray):
        return np.argmax(x, axis=-1)


class StandardScaler(Transform):
    """
    Own implementation of scikit-learn StandardScaler to be used in the project.
    """
    def __init__(self, min_value=None):
        self._mean = None
        self._std = None
        self.min_value = 2 * np.finfo(float).eps if min_value is None else abs(min_value)

    def transform(self, x: np.ndarray):
        self.reset()
        self._mean = np.mean(x, axis=0, keepdims=True)
        self._std = np.std(x, axis=0, keepdims=True)
        self._std[self._std <= self.min_value] = 1   # When std is zero, all the data have the same value
        y = x.copy()
        y -= self._mean
        y /= self._std
        return y

    def inverse_transform(self, x: np.ndarray):
        if self._has_mean_std():
            y = x.copy()
            y *= self._std
            y += self._mean
            self.reset()
            return y
        else:
            raise RuntimeError(f"Unable to apply inverse scaling: "
                               f"mean and standard deviation for original data have been eliminated!")

    def reset(self):
        self._mean = None
        self._std = None

    def _has_mean_std(self):
        return (self._mean is not None) and (self._std is not None)


__all__ = [
    'Transform',
    'OneHotEncoder',
    'StandardScaler',
]
