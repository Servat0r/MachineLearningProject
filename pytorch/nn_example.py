import math
import torch
import torch.nn as tnn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

import nnfs
from nnfs.datasets import sine_data

nnfs.init()


INPUT_DIM = 1
HIDDEN_LAYER_SIZE = 64
TRAIN_BATCH_SIZE = 1000
EVAL_BATCH_SIZE = 2000
TRAIN_MB_SIZE = 1000
N_EPOCHS = 200
SEED = 10
FACTOR = 1.
NOISE_FACTOR = 1.5
MB_NUM = math.ceil(TRAIN_BATCH_SIZE / TRAIN_MB_SIZE)


class NeuralNetwork(tnn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_layer_dim=HIDDEN_LAYER_SIZE):
        super(NeuralNetwork, self).__init__()
        self.flatten = tnn.Flatten()
        self.linear_relu_stack = tnn.Sequential(
            tnn.Linear(input_dim, hidden_layer_dim),
            tnn.ReLU(),
            tnn.Linear(hidden_layer_dim, hidden_layer_dim),
            tnn.ReLU(),
            tnn.Linear(hidden_layer_dim, 1),
            tnn.ReLU(),
        )
        # self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        for layer in self.linear_relu_stack:
            if isinstance(layer, tnn.Linear):
                layer.weight.data.uniform_(-1.0, 1.0)

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Global L1 regularization (credits to: https://stackoverflow.com/questions/42704283/l1-l2-regularization-in-pytorch)
class L1(tnn.Module):
    def __init__(self, module: tnn.Module, weight_decay):
        super().__init__()
        self.module = module
        self.weight_decay = weight_decay

        # Backward hook is registered on the specified module
        self.hook = self.module.register_full_backward_hook(self._weight_decay_hook)

    # Not dependent on backprop incoming values, placeholder
    def _weight_decay_hook(self, *_):
        for param in self.module.parameters():
            # If there is no gradient or it was zeroed out
            # Zeroed out using optimizer.zero_grad() usually
            # Turn on if needed with grad accumulation/more safer way
            if param.grad is None or torch.all(param.grad == 0.0):  # todo check if add reg term however
                # Apply regularization on it
                param.grad = self.regularize(param)

    def regularize(self, parameter):
        # L1 regularization formula
        return self.weight_decay * torch.sign(parameter.data)

    def forward(self, *args, **kwargs):
        # Simply forward and args and kwargs to module
        return self.module(*args, **kwargs)


x_train, y_train = sine_data()
x_train = x_train.reshape(1000, 1, 1)
y_train = y_train.reshape(1000, 1, 1)

simple_nn = NeuralNetwork()
l1_simple_nn = simple_nn # L1(simple_nn, weight_decay=0.001).float()
mse_loss = tnn.MSELoss(reduction='mean')
# optimizer = torch.optim.SGD(l1_simple_nn.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(l1_simple_nn.parameters(), lr=0.005)

train_input_shape = (TRAIN_BATCH_SIZE, 1, INPUT_DIM)
test_input_shape = (EVAL_BATCH_SIZE, 1, INPUT_DIM)

np.random.seed(SEED)
torch.manual_seed(SEED)

# x_train = FACTOR * np.random.randn(TRAIN_BATCH_SIZE, 1, INPUT_DIM)  # 2000 inputs of dimension 100
# y_train = np.sin(2 * np.pi * np.sum(x_train, axis=2)) + 0 * np.random.randn(TRAIN_BATCH_SIZE, 1)
# y_train = np.square(x_train)
# x_train += np.random.randn(TRAIN_BATCH_SIZE, 1, INPUT_DIM)
# y_train = np.reshape(y_train, (TRAIN_BATCH_SIZE, 1, 1))

x_train, y_train = torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()
# y = sin(x1+...+xn) + random (gaussian) noise

x_eval = 100. * np.random.randn(EVAL_BATCH_SIZE, 1, INPUT_DIM)
# y_eval = np.sin(2 * np.pi * np.sum(x_eval, axis=2)) + 0 * np.random.randn(EVAL_BATCH_SIZE, 1)
y_eval = np.square(x_eval)
# x_eval += np.random.randn(EVAL_BATCH_SIZE, 1, INPUT_DIM)
y_eval = np.reshape(y_eval, (EVAL_BATCH_SIZE, 1, 1))

x_eval, y_eval = torch.from_numpy(x_eval).float(), torch.from_numpy(y_eval).float()
# x_eval, y_eval = torch.from_numpy(x_eval), torch.from_numpy(y_eval)


train_dataset, eval_dataset = TensorDataset(x_train, y_train), TensorDataset(x_eval, y_eval)
train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_MB_SIZE, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=EVAL_BATCH_SIZE)


def main_loop():
    total_train_mse_losses = []
    epoch_training_mse_losses = torch.zeros(MB_NUM)

    total_eval_mse_losses = []
    # print(f"[Before Training]: {TRAIN_BATCH_SIZE} training examples of dimension {x_train.shape[2]}")
    for epoch in range(N_EPOCHS):
        train_iter, eval_iter = iter(train_dataloader), iter(eval_dataloader)
        for mb in range(MB_NUM):
            next_x, next_y = next(train_iter)
            optimizer.zero_grad()
            y_hat = l1_simple_nn(next_x)
            mse_losses = mse_loss(y_hat, next_y)
            """
            start, end = mb * MB_NUM, min((mb + 1) * TRAIN_MB_SIZE, TRAIN_BATCH_SIZE)
            y_hat = l1_simple_nn(x_train[start:end])
            mse_losses = mse_loss(y_hat, y_train[start:end])
            acc = torch.abs(y_hat - y_train[start:end])
            """
            print(f"[Epoch {epoch}, Minibatch {mb}]:", f"Average MSE Loss over {TRAIN_MB_SIZE} training examples =",
                  mse_losses.item(), f"Average distance of predicted outputs from real ones =")  #, torch.mean(acc)),
            epoch_training_mse_losses[mb] = mse_losses.item()
            mse_losses.backward()
            optimizer.step()
        total_train_mse_losses.append(torch.mean(epoch_training_mse_losses).item())
        print(f"[Before Evaluating (Epoch {epoch})]: {EVAL_BATCH_SIZE} test examples")
        with torch.no_grad():
            eval_y_hat = l1_simple_nn(x_eval)
            eval_mse_losses = mse_loss(eval_y_hat, y_eval)
            eval_acc = np.array(torch.abs(eval_y_hat - y_eval))
            total_eval_mse_losses.append(np.array(eval_mse_losses).item())
            print(f"[After Evaluating (Epoch {epoch})]:", f"Average MSE Loss over {EVAL_BATCH_SIZE} examples",
                  eval_mse_losses, f"Average distance of predicted outputs from real ones =", np.mean(eval_acc))
    print(
        f"[After Training]: ", f"Average Training and Evaluation MSE Losses per epoch: ",
        str([(trloss, evloss) for trloss, evloss in zip(total_train_mse_losses, total_eval_mse_losses)])
    )


L_TRAIN = 10000
L_TEST = 1000


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
    main_loop()
