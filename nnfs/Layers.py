import numpy as np

from ActivationFunctions import ActivationBase


class LayerBase:
    def __init__(self):
        self.regularization = None
        self.prev = None
        self.next = None
        pass

    def isTrainable(self):
        return False

    # Forward pass
    def forward(self, inputs, training):
        pass

    def backward(self, dvalues):
        pass

    def regularization_loss(self):
        if self.regularization is None:
            return 0.
        else:
            return self.regularization.regularization_loss(self)


class Input(LayerBase):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, inputs, training):
        self.output = inputs


class Dense(LayerBase):
    def __init__(self, n_neurons, *, n_inputs=None, activationFunName=None, regularization=None):
        super().__init__()
        self.regularization = regularization
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs

        if n_inputs is not None:
            self._set_n_inputs(n_inputs)

        self.output = None
        self.dinputs = None
        self.dbiases = None
        self.dweights = None

        if activationFunName is not None:
            self.activation = ActivationBase.GetActivationByName(activationFunName)
        else:
            self.activation = None

    def isTrainable(self):
        return True

    def _set_n_inputs(self, n_inputs):
        self.n_inputs = n_inputs
        # self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.weights = np.random.uniform(-0.7, 0.7, (n_inputs, self.n_neurons))
        # self.biases = np.random.rand(1,n_neurons)
        self.biases = np.zeros((1, self.n_neurons))

    # Forward pass
    def forward(self, inputs, training):
        if self.n_inputs is None:
            self._set_n_inputs(inputs.shape[1])

        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
        if self.activation is not None:
            self.activation.forward(self.output)
            self.output = self.activation.output

    # Backward pass
    def backward(self, dvalues):
        if self.activation is not None:
            self.activation.backward(dvalues)
            dvalues = self.activation.dinputs

        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.regularization is not None:
            self.dweights, self.dbiases = self.regularization.regularization_layer(self)

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
