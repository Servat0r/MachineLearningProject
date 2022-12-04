# Base tests on scikit-learn datasets
from __future__ import annotations
from sklearn.datasets import load_diabetes, fetch_california_housing  # Regression datasets
from sklearn.model_selection import train_test_split

from tests.utils import *
from core.callbacks import TrainingCSVLogger, EarlyStopping, OptimizerMonitor, InteractiveLogger
from core.metrics import MEE, RMSE, Timing
from core.data import ArrayDataset, DataLoader
import core.utils as cu
import core.modules as cm


def get_model_regression(in_size, hidden_sizes, out_size, winit_low=-0.5, winit_high=0.5, l2_lambda=1e-5):
    layers = [cm.Input()]
    sizes = [in_size] + list(hidden_sizes)
    for i in range(len(sizes)-1):
        p, q = sizes[i], sizes[i+1]
        layers.append(
            cm.Dense(p, q, cm.Tanh(), cu.RandomUniformInitializer(winit_low, winit_high),
                     weights_regularizer=cm.L2Regularizer(l2_lambda), biases_regularizer=None)
        )
    layers.append(
        cm.Linear(sizes[-1], out_size, cu.RandomUniformInitializer(winit_low, winit_high),
                  weights_regularizer=None, biases_regularizer=None)
    )
    model = cm.Model(layers)
    return model


def test_diabetes(hidden_sizes, winit_low=-0.1, winit_high=0.1, epoch_shuffle=True):

    # Load and split dataset
    X, y = load_diabetes(return_X_y=True)
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    X_train, X_eval, y_train, y_eval = train_test_split(X_dev, y_dev, test_size=0.3, random_state=42, shuffle=True)

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[-1])
    X_eval = X_eval.reshape(X_eval.shape[0], 1, X_eval.shape[-1])

    y_train = y_train.reshape(y_train.shape[0], 1, 1)
    y_eval = y_eval.reshape(y_eval.shape[0], 1, 1)

    train_dataset, eval_dataset = ArrayDataset(X_train, y_train), ArrayDataset(X_eval, y_eval)
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=epoch_shuffle)
    eval_dataloader = DataLoader(eval_dataset, batch_size=len(eval_dataset))

    # Optimizer and loss
    optimizer = cm.SGD(lr=1e-5, momentum=0.9)
    loss = cm.MSELoss(const=0.5, reduction='mean')

    model = get_model_regression(10, hidden_sizes, 1, winit_low, winit_high, l2_lambda=1e-7)
    model.compile(
        optimizer, loss, metrics=[MEE(), RMSE(), Timing()]
    )
    optim_state = []
    early_stopping = EarlyStopping(monitor='Val_RMSE', min_delta=1e-3, patience=100, return_best_result=True)
    history = model.train(
        train_dataloader, eval_dataloader, max_epochs=1000, callbacks=[
            TrainingCSVLogger(float_round_val=8), OptimizerMonitor(optim_state), early_stopping, InteractiveLogger(),
        ]
    )
    print(f'Training stopped at epoch {len(history)}, '
          f'best model reached at {early_stopping.get_best_epoch()} '
          f'with a value of {early_stopping.get_best_value()} '
          f'(last reference value is = {early_stopping.last_value_recorded} with min delta = {early_stopping.min_delta})')
    plot_history(0, history=history)


def test_california_housing(hidden_sizes, winit_low=-0.1, winit_high=0.1, epoch_shuffle=True):

    # Load and split dataset
    X, y = fetch_california_housing(return_X_y=True)
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    X_train, X_eval, y_train, y_eval = train_test_split(X_dev, y_dev, test_size=0.3, random_state=42, shuffle=True)

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[-1])
    X_eval = X_eval.reshape(X_eval.shape[0], 1, X_eval.shape[-1])

    y_train = y_train.reshape(y_train.shape[0], 1, 1)
    y_eval = y_eval.reshape(y_eval.shape[0], 1, 1)

    train_dataset, eval_dataset = ArrayDataset(X_train, y_train), ArrayDataset(X_eval, y_eval)
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=epoch_shuffle)
    eval_dataloader = DataLoader(eval_dataset, batch_size=len(eval_dataset))

    # Optimizer and loss
    optimizer = cm.SGD(lr=1e-5, momentum=0.9)
    loss = cm.MSELoss(const=0.5, reduction='mean')

    model = get_model_regression(8, hidden_sizes, 1, winit_low, winit_high, l2_lambda=1e-7)
    model.compile(
        optimizer, loss, metrics=[MEE(), RMSE(), Timing()]
    )
    optim_state = []
    early_stopping = EarlyStopping(monitor='Val_RMSE', min_delta=1e-3, patience=100, return_best_result=True)
    history = model.train(
        train_dataloader, eval_dataloader, max_epochs=1000, callbacks=[
            TrainingCSVLogger(float_round_val=8), OptimizerMonitor(optim_state), early_stopping, InteractiveLogger(),
        ]
    )
    print(f'Training stopped at epoch {len(history)}, '
          f'best model reached at {early_stopping.get_best_epoch()} '
          f'with a value of {early_stopping.get_best_value()} '
          f'(last reference value is = {early_stopping.last_value_recorded} with min delta = {early_stopping.min_delta})')
    plot_history(0, history=history)


if __name__ == '__main__':
    test_diabetes(hidden_sizes=(4,))
    # test_california_housing(hidden_sizes=(4,))
    exit(0)
