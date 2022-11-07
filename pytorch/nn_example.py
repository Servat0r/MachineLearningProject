import torch
import torch.nn as tnn
import numpy as np


class NeuralNetwork(tnn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = tnn.Flatten()
        self.linear_relu_stack = tnn.Sequential(
            tnn.Linear(28*28, 512),
            tnn.ReLU(),
            tnn.Linear(512, 512),
            tnn.ReLU(),
            tnn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class Perceptron(tnn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.flatten = tnn.Flatten()
        self.body = tnn.Sequential(
            tnn.Linear(3, 1, bias=False, dtype=torch.float64),
            tnn.Tanh(),
        )

    def forward(self, x):
        # x = self.flatten(x)
        x = torch.from_numpy(x)
        y_hat = self.body(x)
        return y_hat


perceptron = Perceptron()
loss_fn = tnn.MSELoss()
optimizer = torch.optim.SGD(perceptron.parameters(), lr=0.1)

L_TRAIN, L_TEST, INPUT_DIM = 5000, 1000, 3
train_input_shape = (L_TRAIN, INPUT_DIM, 1)
test_input_shape = (L_TEST, INPUT_DIM, 1)

train_inputs = np.random.randn(*train_input_shape)
# train_inputs = np.linspace(-100, 0, L_TRAIN * INPUT_DIM, dtype=np.float64)  # np.random.randn(*train_input_shape)
# train_inputs = np.reshape(train_inputs, train_input_shape)

# train_outputs = np.sign(np.square(np.sum(train_inputs, axis=1)))  # + np.random.randn(L_TRAIN, 1))
train_outputs = np.sign(np.sin(np.sum(train_inputs, axis=1)))
# train_outputs = np.sign(np.sum(train_inputs, axis=1))

test_inputs = np.random.randn(*test_input_shape) # .uniform(-math.pi, math.pi, test_input_shape)
# test_inputs = np.linspace(0, 100, L_TEST * INPUT_DIM, dtype=np.float64)
# test_inputs = np.reshape(test_inputs, test_input_shape)

# test_outputs = np.sign(np.square(np.sum(test_inputs, axis=1)))  # + np.random.randn(L_TEST, 1))
test_outputs = np.sign(np.sin(np.sum(test_inputs, axis=1)))
# test_outputs = np.sign(np.sum(test_inputs, axis=1))


def train_loop(model, train_inputs, train_outputs, loss_fn, optimizer):
    acc = 0
    for i in range(L_TRAIN):
        x, y = train_inputs[i], torch.from_numpy(train_outputs[i])
        # Compute prediction and loss
        x = np.reshape(x, (1, INPUT_DIM))
        y_hat = model(x)
        pred = torch.sign(y_hat)
        loss = loss_fn(pred, y)
        if pred.item() == y:
            acc += 1
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc = acc / L_TRAIN
    print(f'Average train accuracy: {acc}')


def test_loop(model, test_inputs, test_outputs, loss_fn):
    test_loss, correct = 0, 0
    with torch.no_grad():
        for i in range(L_TEST):
            x, y = test_inputs[i], torch.from_numpy(test_outputs[i])
            x = np.reshape(x, (1, INPUT_DIM))
            pred = torch.sign(model(x))
            test_loss += loss_fn(pred, y).item()
            if pred.item() == y:
                correct += 1

    test_loss /= L_TEST
    correct /= L_TEST
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    train_loop(perceptron, train_inputs, train_outputs, loss_fn, optimizer)
    test_loop(perceptron, test_inputs, test_outputs, loss_fn)