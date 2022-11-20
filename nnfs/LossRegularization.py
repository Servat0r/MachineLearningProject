import numpy as np


class RegL1:
    def __init__(self, *,kernel_regularizer, bias_regularizer):
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

    # Regularization loss calculation
    def regularization_loss(self, layer):
        # 0 by default
        regularization_loss = 0

        # L1 regularization - weights
        # calculate only when factor greater than 0
        if self.kernel_regularizer > 0:
            regularization_loss += self.kernel_regularizer * np.sum(np.abs(layer.weights))

        # L1 regularization - biases
        # calculate only when factor greater than 0
        if self.bias_regularizer > 0:
            regularization_loss += self.bias_regularizer * np.sum(np.abs(layer.biases))

        return regularization_loss

    def regularization_layer(self, layer):
        dweights = layer.dweights.copy()
        dbiases = layer.dbiases.copy()

        # Gradients on regularization
        # L1 on weights
        if self.kernel_regularizer > 0:
            dL1 = np.ones_like(layer.weights)
            dL1[layer.weights < 0] = -1
            dweights += self.kernel_regularizer * dL1

        # L1 on biases
        if (self.bias_regularizer>0):
            dL1 = np.ones_like(layer.biases)
            dL1[layer.biases < 0] = -1
            dbiases += self.bias_regularizer * dL1

        return dweights, dbiases


