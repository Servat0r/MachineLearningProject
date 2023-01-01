# Dataloaders for batching datasets, shuffling and training/evaluation cycles handling
from __future__ import annotations
from ..utils import *
from .datasets import *
import math
import json
from time import perf_counter


class DataLoader(Iterable):
    """
    The DataLoader class provides a convenient way to iterate over a dataset according
    to a given batch size and to handle random shuffling of the dataset at the end of
    each epoch. A DataLoader for training (and optionally validation) data is required
    in Model.train().
    """

    @property
    def logger_name(self):
        return f"{type(self).__name__}__{self.log_to}" if self.log_to is not None else None

    def __init__(
            self, dataset: BaseDataset, batch_size: int = 1, shuffle: bool = False,
            shuffle_seed: int = 0, log_to: str = None, log_values: bool = False,
    ):
        """
        :param dataset: The dataset to use for retrieving samples.
        :param batch_size: Size of each (mini)batch of data to retrieve from the dataset.
        Defaults to 1.
        :param shuffle: Set to True for shuffling the dataset (actually, the way the dataloder
        accesses dataset items) at the end of each epoch. Defaults to False.
        :param log_to: If not None, dataloder will log each epoch permutation, (mini)batch indices
        and retrieval times to the file specified by log_to; intended mainly for debugging purposes.
        Defaults to None.
        :param log_values: If True, dataloader will log indices per minibatch for each epoch.
        Defaults to False.
        """
        self.dataset = dataset
        self.batch_size = len(dataset) if batch_size is None else batch_size
        self.shuffle_every_epoch = shuffle
        if log_to is not None:
            self.log_to = open(log_to, 'w')
        else:
            self.log_to = None
            self.logger = None
            self.log_handler = None
        self.log_values = log_values
        self.permutation = None
        self.current_index = 0  # used for indexing on the permutation
        self.current_time = None   # used for timing dataset access
        self.rng = npr.default_rng(seed=shuffle_seed) if self.shuffle_every_epoch is not None else None
        self.current_log_message = []
        self.current_epoch = 0

    def is_logging(self):
        return self.log_to is not None

    def next_minibatch_indices(self):
        left = self.current_index
        right = min(left + self.batch_size, len(self.dataset))
        return left, right

    def get_batch_number(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def before_cycle(self):
        """
        Callback to use before any training cycle (e.g., to reset internal state of the dataloader).
        """
        self.permutation = np.array(range(len(self.dataset)))   # initially, we start with all elements "in order"
        self.current_index = 0
        self.current_time = None

    def after_cycle(self):
        """
        Callback to use after having finished a training cycle.
        """
        self.permutation = None
        self.current_index = None
        self.current_time = None
        if self.is_logging():
            json.dump(self.current_log_message, self.log_to, indent=4)
            self.log_to.close()
            self.log_to = None
            self.logger = None
            self.log_handler = None
        self.current_log_message = []
        self.current_epoch = 0

    def before_epoch(self):
        """
        Callback to use before each training epoch.
        """
        if self.permutation is None:
            self.permutation = np.array(range(len(self.dataset)))
        self.current_index = 0
        if self.is_logging():
            self.current_log_message.append({
                'epoch': self.current_epoch,
                'minibatches': []
            })

    def after_epoch(self):
        """
        Callback to use after each training epoch.
        """
        if self.shuffle_every_epoch:
            # self.rng.permuted(self.permutation, axis=0, out=self.permutation)
            self.permutation = self.rng.permuted(self.permutation, axis=0)
        self.current_epoch += 1

    def before_minibatch(self):
        """
        Callback to use before each minibatch training.
        """
        left, right = self.next_minibatch_indices()
        if self.is_logging():
            if self.log_values:
                self.current_log_message[self.current_epoch]['minibatches'].append({
                    'values': self.permutation[left:right].tolist(),
                })
            else:
                self.current_log_message[self.current_epoch]['minibatches'].append({})
            self.current_time = perf_counter()

    def after_minibatch(self):
        """
        Callback to use after each minibatch training.
        """
        if self.is_logging():
            self.current_time = perf_counter() - self.current_time
            self.current_log_message[self.current_epoch]['minibatches'][-1]['time'] = self.current_time
        left, right = self.next_minibatch_indices()
        self.current_index = right

    def __next__(self):
        if self.current_index >= len(self.permutation):    # End of the dataset for current epoch
            return None
        else:
            self.before_minibatch()
            left, right = self.next_minibatch_indices()
            data = self.dataset.get_batch(indices=self.permutation[left:right])
            self.after_minibatch()
            return data

    def __iter__(self):
        return self


__all__ = [
    'DataLoader',
]
