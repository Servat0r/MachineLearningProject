# Datasets for usage with Models
from __future__ import annotations
from ..utils import *
import os
import cv2
import operator


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
        :param x: Input data; first shape value is always treated as number of elements.
        :param y: Input labels first shape value is always treated as number of elements
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


class ImageFromDirectoryDataset(BaseDataset):

    def __init__(self, path: str, labels_file: str, img_width: int, img_height: int):
        self.base_path = path
        with open(labels_file, 'r') as lb:
            fdata = lb.readlines()
            fdata = np.array([int(fline.strip()) for fline in fdata])
            self.labels = fdata
        self.length = len(self.labels)
        self.dir_content = [fpath.strip() for fpath in os.listdir(self.base_path) if not os.path.isdir(fpath)]
        self.img_width = img_width
        self.img_height = img_height

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        fname = self.dir_content[item]
        return cv2.imread(fname), self.labels[item]

    def get_batch(self, indices: int | Iterable[int], out_labels: np.ndarray = None):
        fnames = operator.itemgetter(*indices)(self.dir_content)
        images = np.zeros((len(fnames), 1, self.img_width * self.img_height))
        if out_labels is None:
            out_labels = np.take(self.labels, indices)
        else:
            np.take(self.labels, indices, out=out_labels)
        for i in range(len(fnames)):
            fname = fnames[i]
            images[i, :] = cv2.imread(fname)[:, :]
        return images, out_labels


__all__ = [
    'BaseDataset',
    'ArrayDataset',
]
