# Commons datasets and dataloaders: MONKs, ML-CUP, MNIST
from __future__ import annotations
import os
import gzip
from ..utils import *
from .datasets import *
from .dataloading import *


class MNIST(BaseDataset):

    IMAGE_SIZE = 28

    def __init__(
            self, base_dir: str, train: bool = True, flatten: bool = False,
    ):  # todo add also downlaod
        self.base_dir = base_dir
        self.train = train
        if train:
            img_folder = os.path.join(self.base_dir, 'train-images-idx3-ubyte.gz')
            labels_folder = os.path.join(self.base_dir, 'train-labels-idx1-ubyte.gz')
            self.images = gzip.open(img_folder, 'r')
            self.labels = gzip.open(labels_folder, 'r')
        else:
            img_folder = os.path.join(self.base_dir, 't10k-images-idx3-ubyte.gz')
            labels_folder = os.path.join(self.base_dir, 't10k-labels-idx1-ubyte.gz')
            self.images = gzip.open(img_folder, 'r')
            self.labels = gzip.open(labels_folder, 'r')
        self.images.read(16)
        self.labels.read(8)    # todo check!
        # Read image data into numpy arrays
        buf = self.images.read(self.IMAGE_SIZE * self.IMAGE_SIZE * len(self))
        self.image_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        if flatten:
            self.image_data = self.image_data.reshape((len(self), 1, self.IMAGE_SIZE ** 2))
        self.image_data = self.image_data.reshape((len(self), self.IMAGE_SIZE, self.IMAGE_SIZE))
        # Do the same for labels data
        buf = self.labels.read(len(self))
        self.labels_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)

    def __len__(self):
        return 60_000 if self.train else 10_000

    def __getitem__(self, item):
        return self.image_data[item], self.labels_data[item]

    def get_batch(self, indices: int | Iterable[int]):
        return np.take(self.image_data, indices, axis=0), np.take(self.labels_data, indices, axis=0)


__all__ = [
    'MNIST',
]
