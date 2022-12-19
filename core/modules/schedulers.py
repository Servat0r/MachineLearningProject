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

    # An example of linear learning_rate decay that can be given as argument when initializing
    def __init__(self, start_value: float, end_value: float, max_iterations: int, round_value: int = None):
        self.start_value = start_value
        self.end_value = end_value
        self.max_iterations = max_iterations
        self.round_value = round_value

    def __call__(self, iteration: int, current_value: float):
        beta = iteration / self.max_iterations
        result = (1. - beta) * self.start_value + beta * self.end_value
        if self.round_value is not None:
            result = round(result, self.round_value)
        return result

    def __eq__(self, other):
        if not isinstance(other, LinearDecayScheduler):
            return False
        return all([
            self.start_value == other.start_value, self.end_value == other.end_value,
            self.max_iterations == other.max_iterations, self.round_value == other.round_value,
        ])


class IterBasedDecayScheduler(Scheduler):

    def __init__(self, start_value: float, decay: float, round_val: int = None):
        self.decay = decay
        self.start_value = start_value
        self.round_value = round_val

    def __call__(self, iteration: int, current_value: float):
        result = self.start_value / (1. + self.decay * iteration)
        if self.round_value is not None:
            result = round(result, self.round_value)
        return result

    def __eq__(self, other):
        if not isinstance(other, IterBasedDecayScheduler):
            return False
        return all([
            self.decay == other.decay, self.start_value == other.start_value, self.round_value == other.round_value
        ])


class ExponentialDecayScheduler(Scheduler):

    def __init__(self, start_value: float, alpha: float, round_val: int = None):
        self.start_value = start_value
        self.alpha = alpha
        self.round_value = round_val

    def __call__(self, iteration: int, current_value: float):
        result = self.start_value * np.exp(-1.0 * self.alpha * iteration)
        if self.round_value is not None:
            result = round(result, self.round_value)
        return result

    def __eq__(self, other):
        if not isinstance(other, ExponentialDecayScheduler):
            return False
        return all([
            self.start_value == other.start_value, self.alpha == other.alpha, self.round_value == other.round_value,
        ])


__all__ = [
    'Scheduler',
    'LinearDecayScheduler',
    'IterBasedDecayScheduler',
    'ExponentialDecayScheduler',
]
