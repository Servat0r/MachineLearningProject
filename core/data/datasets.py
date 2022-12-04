# Datasets for usage with Models
from __future__ import annotations
from ..utils import *


class BaseDataset:

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def get_batch(self, indices: int | Iterable[int]):
        pass


class ArrayDataset(BaseDataset):
    """
    Simple dataset from a set of arrays already in memory.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        :param x: Input data. First shape value is always treated as number of elements.
        :param y: Input labels. First shape value is always treated as number of elements
        and must be equal to the correspondent for x.
        """
        x_shape, y_shape = x.shape, y.shape
        length = x_shape[0]
        if not length == y_shape[0]:
            raise ValueError(
                f"Input data and labels array do not have the same batch dimension: "
                f"the first has {x_shape[0]}, while the second has {y_shape[0]}"
            )
        self.x = x
        self.y = y
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def get_batch(self, indices: int | Iterable[int], out_x: np.ndarray = None, out_y: np.ndarray = None):
        return np.take(self.x, indices, axis=0, out=out_x), np.take(self.y, indices, axis=0, out=out_y)


__all__ = [
    'BaseDataset',
    'ArrayDataset',
]
