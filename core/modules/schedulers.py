# Schedulers for adjusting Learning Rate (inspired by PyTorch)
# Available ones are for expontential, polynomial and linear decay
# Each learning rate is assumed fixed unless a particular scheduler is passed for it.
from __future__ import annotations
from ..utils import *


class Scheduler(Callable):
    """
    Base class for schedulers. A Scheduler is in general
    a Callable that is used to have a dynamic learning
    rate, and in particular it is updated w.r.t. current
    iteration (i.e., epoch) and value.
    """

    @abstractmethod
    def __call__(self, iteration: int, current_value: float):
        pass


class LinearDecayScheduler(Scheduler):
    """
    Linear learning_rate decay according to the formula:
     - lr(x) := (1 - x/N) * start_value + x/N * end_value (if round_value = None)
     - lr(x) := round((1 - x/N) * start_value + x/N * end_value, round_value) (otherwise)
    where x is the current epoch, N is the maximum number of epochs,
    start_value and end_value are initial and final learning rates.
    """

    def __init__(self, start_value: float, end_value: float, max_iterations: int, round_value: int = None):
        self.start_value = start_value
        self.end_value = end_value
        self.max_iterations = max_iterations
        self.round_value = round_value

    def __call__(self, iteration: int, current_value: float):
        if iteration < self.max_iterations:
            beta = iteration / self.max_iterations
            result = (1. - beta) * self.start_value + beta * self.end_value
        else:
            result = self.end_value
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
    """
    A learning rate decay strategy that implements the function:
    - lr(x) := start_value / (1 + decay * x) (if round_val = None)
    - lr(x) := round(start_value / (1 + decay * x), round_val) (otherwise)
    where x is the current epoch, decay is a constant and start_value
    is the initial learning rate.
    """

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
    """
    A Scheduler that implements the function:
    lr(x) := start_value * e^(-alpha*x) (if round_val is None)
    lr(x) := round(start_value * e^(-alpha*x), round_val) (otherwise)
    where x is the current epoch, alpha is a constant and
    start_value is the initial learning rate.
    """

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
