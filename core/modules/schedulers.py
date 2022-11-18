# Schedulers for adjusting Learning Rate (inspired by PyTorch)
# Available ones are for expontential, polynomial and linear decay
# Each learning rate is assumed fixed unless a particular scheduler is passed for it.
from __future__ import annotations
from ..utils import *


class Scheduler(Callable):

    @abstractmethod
    def __call__(self, iteration: int, current_value: float):
        pass


class LinearDecayScheduler(Scheduler):

    # An example of linear lr decay that can be given as argument when initializing
    def __init__(self, start_value: float, end_value: float, max_iter: int):
        self.start_value = start_value
        self.end_value = end_value
        self.max_iter = max_iter

    def __call__(self, iteration: int, current_value: float):
        beta = iteration / self.max_iter
        return (1. - beta) * self.start_value + beta * self.end_value


class IterBasedDecayScheduler(Scheduler):   # todo I don't remember the actual name

    def __init__(self, decay: float, start_value: float):
        self.decay = decay
        self.start_value = start_value

    def __call__(self, iteration: int, current_value: float):
        return self.start_value / (1. + self.decay * iteration)


class ExponentialDecayScheduler(Scheduler):

    def __init__(self, start_value: float, alpha: float):
        self.start_value = start_value
        self.alpha = alpha

    def __call__(self, iteration: int, current_value: float):
        return self.start_value * np.exp(-1.0 * self.alpha * iteration)  # todo is float64 -> float conversion implicit?


__all__ = [
    'Scheduler',
    'LinearDecayScheduler',
    'IterBasedDecayScheduler',
    'ExponentialDecayScheduler',
]
