from core.utils import np
from core.data import ArrayDataset
import matplotlib.pyplot as plt
import core.utils as cu
import core.modules as cm
from core.data import DataLoader
from core.callbacks import Callback


INPUT_DIM = 8
HIDDEN_SIZE = 4
OUTPUT_DIM = 2
N_SAMPLES = 1000
SEED = 20
# Default start values for train and eval in arange_* functions.
TRAIN_START = 0
EVAL_START = N_SAMPLES * INPUT_DIM

np.random.seed(SEED)


# Create dataset
# ---------------------- ARANGE-BASED GENERATORS ----------------------------------
def arange_sine_data(samples=1000, input_dim=1, output_dim=1, start=TRAIN_START):
    stop = samples * input_dim + start
    x = np.arange(start=start, stop=stop).reshape((samples, 1, input_dim)) / stop
    y = np.sin(np.sum(x, axis=-1))
    z = np.zeros((samples, output_dim))
    for i in range(samples):
        for j in range(output_dim):
            z[i, j] = y[i]
    z = z.reshape((samples, 1, output_dim))
    return x, z


def arange_square_data(samples=1000, input_dim=1, output_dim=1, start=TRAIN_START):
    stop = samples * input_dim + start
    x = np.arange(start=start, stop=stop).reshape((samples, 1, input_dim)) / stop
    y = np.square(np.sum(x, axis=-1))
    z = np.zeros((samples, output_dim))
    for i in range(samples):
        for j in range(output_dim):
            z[i, j] = y[i]
    z = z.reshape((samples, 1, output_dim))
    return x, z


def arange_sqrt_data(samples=1000, input_dim=1, output_dim=1, start=TRAIN_START):
    stop = samples * input_dim + start
    x = np.arange(start=start, stop=stop).reshape((samples, 1, input_dim)) / stop
    y = np.sqrt(np.abs(np.sum(x, axis=-1)))
    z = np.zeros((samples, output_dim))
    for i in range(samples):
        for j in range(output_dim):
            z[i, j] = y[i]
    z = z.reshape((samples, 1, output_dim))
    return x, z


# ----------------------- NORMAL DISTRIBUTION - BASED GENERATORS ------------------
def randn_sine_data(samples=1000, input_dim=1, output_dim=1, normalize=True):
    x = np.random.randn(samples * input_dim).reshape((samples, 1, input_dim))
    if normalize:
        x /= np.max(x)  # normalization
    y = np.sin(np.sum(x, axis=-1))
    z = np.zeros((samples, output_dim))
    for i in range(samples):
        for j in range(output_dim):
            z[i, j] = y[i]
    z = z.reshape((samples, 1, output_dim))
    return x, z


def randn_square_data(samples=1000, input_dim=1, output_dim=1, normalize=True):
    x = np.random.randn(samples * input_dim).reshape((samples, 1, input_dim))
    if normalize:
        x /= np.max(x)  # normalization
    y = np.square(np.sum(x, axis=-1))
    z = np.zeros((samples, output_dim))
    for i in range(samples):
        for j in range(output_dim):
            z[i, j] = y[i]
    z = z.reshape((samples, 1, output_dim))
    return x, z


def randn_sqrt_data(samples=1000, input_dim=1, output_dim=1, normalize=True):
    x = np.random.randn(samples * input_dim).reshape((samples, 1, input_dim))
    if normalize:
        x /= np.max(x)  # normalization
    y = np.sqrt(np.abs(np.sum(x, axis=-1)))
    z = np.zeros((samples, output_dim))
    for i in range(samples):
        for j in range(output_dim):
            z[i, j] = y[i]
    z = z.reshape((samples, 1, output_dim))
    return x, z


def generate_dataset(func, samples=N_SAMPLES, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, *args, **kwargs):

    args = () if args is None else args
    kwargs = {} if kwargs is None else kwargs
    x, y = func(samples=samples, input_dim=input_dim, output_dim=output_dim, *args, **kwargs)
    x = x.reshape((x.shape[0], x.shape[1], INPUT_DIM))
    y = y.reshape((y.shape[0], y.shape[1], OUTPUT_DIM))
    train_dataset = ArrayDataset(x, y)

    return x, y, train_dataset


def generate_dataset_and_model(
        func, func_args, eval_samples, mb_size, epoch_shuffle, winit_low, winit_high, l1_lambda, l2_lambda
):
    # Generate train and validation dataset
    x, y, train_dataset = generate_dataset(func)
    func_args = {} if func_args is None else func_args
    x_eval, y_eval, eval_dataset = generate_dataset(
        func, samples=eval_samples, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, **func_args
    )

    # Generate dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=mb_size, shuffle=epoch_shuffle)
    eval_dataloader = DataLoader(eval_dataset, batch_size=N_SAMPLES//5)
    model = cm.Model(
        generate_layers(
            low=winit_low, high=winit_high,
            weights_regularizer=cm.L1L2Regularizer(l1_lambda=l1_lambda, l2_lambda=l2_lambda),
            biases_regularizer=cm.L1L2Regularizer(l1_lambda=l1_lambda, l2_lambda=l2_lambda),
        )
    )
    return train_dataloader, eval_dataloader, model


def plot_losses(start_epoch, train_epoch_losses, eval_epoch_losses=None, other_metric_logs: dict = None):
    # plot training and validation losses
    epochs = np.arange(start_epoch, len(train_epoch_losses))
    plt.plot(epochs, train_epoch_losses[start_epoch:], label='loss')
    if eval_epoch_losses is not None:
        plt.plot(epochs, eval_epoch_losses[start_epoch:], label='val_loss')
    other_metric_logs = {} if other_metric_logs is None else other_metric_logs
    for metric_name, metric_vals in other_metric_logs.items():
        plt.plot(epochs, metric_vals[start_epoch:], label=metric_name)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Train / Eval Losses')
    plt.show()


def plot_history(start_epoch, eval_epoch_losses=None, history=None):
    n_epochs = len(history)
    if start_epoch >= n_epochs:
        raise ValueError(
            f"Starting epoch is higher than the actual number of epochs for which history has valid data:"
            f" expected < {n_epochs}, got {start_epoch}"
        )
    epochs = np.arange(start_epoch, n_epochs)
    """
    if eval_epoch_losses is not None:
        plt.plot(epochs, eval_epoch_losses[start_epoch:], label='val_loss')
    """
    for metric_name, metric_vals in history.items():
        # if metric_name.startswith('Val_'):
        #     continue
        plt.plot(epochs, metric_vals[start_epoch:n_epochs], label=metric_name)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.show()


def generate_layers(low=-0.5, high=0.5, weights_regularizer=None, biases_regularizer=None, grad_reduction='mean'):
    dense1 = cm.Dense(
            INPUT_DIM, HIDDEN_SIZE, cm.Tanh(),
            weights_initializer=cu.RandomUniformInitializer(low, high), grad_reduction=grad_reduction,
            # biases_initializer=cu.RandomUniformInitializer(-1.0, 1.0),
            weights_regularizer=weights_regularizer, biases_regularizer=biases_regularizer,
        )
    dense2 = cm.Dense(
        HIDDEN_SIZE, HIDDEN_SIZE, cm.Tanh(),
        weights_initializer=cu.RandomUniformInitializer(low, high), grad_reduction=grad_reduction,
        # biases_initializer=cu.RandomUniformInitializer(-1.0, 1.0),
        weights_regularizer=weights_regularizer, biases_regularizer=biases_regularizer,
    )
    linear3 = cm.Linear(
        HIDDEN_SIZE, OUTPUT_DIM,
        weights_initializer=cu.RandomUniformInitializer(low, high), grad_reduction=grad_reduction,
        # biases_initializer=cu.RandomUniformInitializer(-1.0, 1.0),
        weights_regularizer=weights_regularizer, biases_regularizer=biases_regularizer,
    )
    return [dense1, dense2, linear3]


class WaitKey(Callback):
    """
    Utility callback for pausing after a training epoch.
    """
    def __init__(self, wait_every=1, prompt=None):
        self.wait_every = wait_every
        self.prompt = '' if prompt is None else prompt

    def after_training_epoch(self, model, epoch, logs=None):
        if epoch % self.wait_every == 0:
            input(self.prompt)
