# Parameters Classes for usage with Optimizers (inspired by PyTorch)
from __future__ import annotations

import numpy as np

from ..utils import *
from .layers import *


class Parameters:
    """
    A parameter (or set of parameters)
    """
    pass


class WeightedLayerParameters(Parameters):

    SUM = 'sum'  # sum all gradient values in the current batch as raw gradient reduction
    MEAN = 'mean'  # take average of all gradient vales in the current batch as raw gradient reduction
    REDS = [SUM, MEAN]

    def __init__(self, weights: np.ndarray, biases: np.ndarray, grad_reduction=None,
                 layer: Layer = None, regularizers=None):
        self.weights, self.biases = weights, biases
        # Gradients for weights updating (initialized to empty values because we overwrite them,
        # otherwise we need to give up on maintaining batch results for each input separate)
        # (i.e., if we want to do something different from summing up over the batches, this
        # would not be possible)
        if not self._check_grad_reduction(grad_reduction):
            raise ValueError(f"Unknown raw gradients reduction '{grad_reduction}'")
        if grad_reduction in self.REDS:
            self.dweights, self.dbiases = np.zeros_like(weights), np.zeros_like(biases)
        else:
            self.dweights, self.dbiases = None, None
        self.layer = layer  # Pointer to referring layer
        self.grad_reduction = grad_reduction
        self.regularizer_updates = {}
        if regularizers is not None:
            regularizers = {regularizers} if not isinstance(regularizers, Iterable) else regularizers
            for regularizer in regularizers:
                regularizer.init_new_parameters({self})
        self.weight_momentums, self.bias_momentums = None, None

    @staticmethod
    def _check_grad_reduction(grad_reduction):
        return (grad_reduction is None) or (grad_reduction == WeightedLayerParameters.SUM) or \
               (grad_reduction == WeightedLayerParameters.MEAN)

    def apply_reduction(self, x: np.ndarray):
        if self.grad_reduction == self.SUM:
            return np.sum(x, axis=0)
        elif self.grad_reduction == self.MEAN:
            return np.mean(x, axis=0)
        else:
            return x

    def get_weights(self, copy=True):
        return self.weights.copy() if copy else self.weights

    def get_biases(self, copy=True):
        return self.biases.copy() if copy else self.biases

    def get_dweights(self, copy=True):
        return self.dweights.copy() if copy else self.weights

    def get_dbiases(self, copy=True):
        return self.dbiases.copy() if copy else self.biases

    def update_grads(self, dweights: np.ndarray, dbiases: np.ndarray):
        # Apply reduction on raw gradients if requested at initialization
        dweights, dbiases = self.apply_reduction(dweights), self.apply_reduction(dbiases)

        self.dweights = dweights
        self.dbiases = dbiases

    def update_weights_and_biases(self, w_vals: np.ndarray, b_vals: np.ndarray):
        self.weights += w_vals
        self.biases += b_vals

    def zero_grads(self):
        """
        Zeroes dweights, dbiases for more security.
        """
        if self.dweights is not None:
            self.dweights.fill(0.)
        if self.dbiases is not None:
            self.dbiases.fill(0.)
        for reg_name, reg_update in self.regularizer_updates.items():
            regw, regb = self.regularizer_updates[reg_name].get('weights'), self.regularizer_updates[reg_name].get('biases')
            if regw is not None:
                regw.fill(0.)
            if regb is not None:
                regb.fill(0.)


__all__ = [
    'Parameters',
    'WeightedLayerParameters',
]
