import numpy as np

class ActivationBase:
    def  __init__(self):
        pass

    # Forward pass
    def forward(self, inputs):
        pass

    # Backward pass
    def backward(self, dvalues):
        pass

    @staticmethod
    def GetActivationByName(activationName):
        if activationName.lower() == "relu":
            return ReLU()
        elif activationName.lower() == "tanh":
            return TanH()
        else:
            raise NotImplementedError("Unexpected activation function")



class ReLU(ActivationBase):

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


# https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
class TanH(ActivationBase):
    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.tanh(inputs)

    #https://discuss.pytorch.org/t/what-is-pytorchs-backwards-function-for-tanh/119389/4
    def backward(self, dvalues):
        # f'(x) = 1-f(x)^2
        self.dinputs = (1.0 - np.tanh(self.inputs) ** 2.0) * dvalues


