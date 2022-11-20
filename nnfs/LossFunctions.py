import numpy as np


class LossBase:
    def __init__(self):
        self.accumulated_sum = None
        self.accumulated_count = None

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def forward(self, y_pred, y_true):
        pass

    def backward(self, dvalues, y_true):
        pass

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y, layers):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        # Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        if isinstance(sample_losses, list):
            self.accumulated_count += len(sample_losses)
        else:
            self.accumulated_count += 1

        regularization_loss = 0
        for layer in layers:
            if layer.isTrainable():
                regularization_loss += layer.regularization_loss()
        return data_loss, regularization_loss

    # Calculates accumulated loss
    def calculate_accumulated(self, layers):

        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count

        regularization_loss = 0
        for layer in layers:
            if layer.isTrainable():
                regularization_loss += layer.regularization_loss()

        # Return the data and regularization losses
        return data_loss, regularization_loss
    @staticmethod
    def GetLossByName(lossName):
        if lossName.lower() == "mse":
            return MeanSquaredError()
        else:
            raise NotImplementedError("Unexpected loss function")


# Mean Squared Error loss
class MeanSquaredError(LossBase):  # L2 loss

    # Forward pass
    def forward(self, y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean((y_true - y_pred) ** 2, axis=-1)

        data_loss = np.mean(sample_losses)
        # Return losses
        return data_loss

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples
