# Base tests on scikit-learn datasets
from __future__ import annotations
from sklearn.datasets import load_iris, load_breast_cancer, fetch_covtype  # Classification datasets
from sklearn.model_selection import train_test_split

from tests.utils import *
from core.callbacks import TrainingCSVLogger, EarlyStopping, OptimizerMonitor, InteractiveLogger
from core.metrics import SparseCategoricalAccuracy
from core.data import ArrayDataset, DataLoader
import core.utils as cu
import core.modules as cm
from core.transforms import StandardScaler


def get_model_classification(in_size, hidden_sizes, out_size, winit_low=-0.5, winit_high=0.5, l2_lambda=1e-5):
    layers = [cm.Input()]
    sizes = [in_size] + list(hidden_sizes)
    for i in range(len(sizes)-1):
        p, q = sizes[i], sizes[i+1]
        layers.append(
            cm.Dense(p, q, cm.Tanh(), cu.RandomUniformInitializer(winit_low, winit_high),
                     weights_regularizer=cm.L2Regularizer(l2_lambda), biases_regularizer=None,
                     gradients_reduction='mean')
        )
    layers.append(
        cm.Linear(sizes[-1], out_size, cu.RandomUniformInitializer(winit_low, winit_high),
                  weights_regularizer=cm.L2Regularizer(l2_lambda), biases_regularizer=None,
                  gradients_reduction='mean')
    )
    layers.append(cm.SoftmaxLayer())
    model = cm.Model(layers)
    return model


def test_iris(hidden_sizes, winit_low=-0.1, winit_high=0.1, epoch_shuffle=True):

    # Load and split dataset
    X, y = load_iris(return_X_y=True)
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
    X_train, X_eval, y_train, y_eval = X_dev, X_test, y_dev, y_test
    # X_train, X_eval, y_train, y_eval = train_test_split(X_dev, y_dev, test_size=0.3, random_state=42, shuffle=True)

    sc = StandardScaler()
    X_train, X_eval = sc.transform(X_train), sc.transform(X_eval)

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[-1])
    X_eval = X_eval.reshape(X_eval.shape[0], 1, X_eval.shape[-1])

    # todo for future, transform them in one-hot encoded versions
    # y_train = y_train.reshape(y_train.shape[0], 1, 1)
    # y_eval = y_eval.reshape(y_eval.shape[0], 1, 1)

    train_dataset, eval_dataset = ArrayDataset(X_train, y_train), ArrayDataset(X_eval, y_eval)
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=epoch_shuffle)
    eval_dataloader = DataLoader(eval_dataset, batch_size=len(eval_dataset))

    # Optimizer and loss
    optimizer = cm.SGD(lr=1e-3, momentum=0.9)
    loss = cm.CrossEntropyLoss(clip_value=1e-7, reduction='mean')
    # loss = cm.MSELoss(const=0.5, reduction='mean')

    model = get_model_classification(4, hidden_sizes, 3, winit_low, winit_high, l2_lambda=1e-7)
    # get_model_regression(10, hidden_sizes, 1, winit_low, winit_high, l2_lambda=1e-7)
    model.compile(
        optimizer, loss, metrics=[SparseCategoricalAccuracy()]
    )
    optim_state = []
    early_stopping = EarlyStopping(
        monitor='Val_SparseCategoricalAccuracy', min_delta=1e-6, patience=100, mode='max', return_best_result=True
    )
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


def test_breast_cancer(hidden_sizes, winit_low=-0.1, winit_high=0.1, epoch_shuffle=True):

    # Load and split dataset
    X, y = load_breast_cancer(return_X_y=True)
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
    X_train, X_eval, y_train, y_eval = X_dev, X_test, y_dev, y_test
    # X_train, X_eval, y_train, y_eval = train_test_split(X_dev, y_dev, test_size=0.3, random_state=42, shuffle=True)

    sc = StandardScaler()
    X_train, X_eval = sc.transform(X_train), sc.transform(X_eval)

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[-1])
    X_eval = X_eval.reshape(X_eval.shape[0], 1, X_eval.shape[-1])

    # todo for future, transform them in one-hot encoded versions
    # y_train = y_train.reshape(y_train.shape[0], 1, 1)
    # y_eval = y_eval.reshape(y_eval.shape[0], 1, 1)

    train_dataset, eval_dataset = ArrayDataset(X_train, y_train), ArrayDataset(X_eval, y_eval)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=epoch_shuffle)
    eval_dataloader = DataLoader(eval_dataset, batch_size=len(eval_dataset))

    # Optimizer and loss
    optimizer = cm.SGD(lr=1e-4)  # , momentum=0.9)
    loss = cm.CrossEntropyLoss(clip_value=1e-7, reduction='mean')
    # loss = cm.MSELoss(const=0.5, reduction='mean')

    model = get_model_classification(30, hidden_sizes, 2, winit_low, winit_high)  # , l2_lambda=1e-7)
    # get_model_regression(10, hidden_sizes, 1, winit_low, winit_high, l2_lambda=1e-7)
    model.compile(
        optimizer, loss, metrics=[SparseCategoricalAccuracy()]
    )
    optim_state = []
    early_stopping = EarlyStopping(
        monitor='Val_SparseCategoricalAccuracy', min_delta=1e-6, patience=100, mode='max', return_best_result=True
    )
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


def test_covtype(hidden_sizes, winit_low=-0.1, winit_high=0.1, epoch_shuffle=True):

    # Load and split dataset
    X, y = fetch_covtype(return_X_y=True, shuffle=True)
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
    X_train, X_eval, y_train, y_eval = X_dev, X_test, y_dev, y_test
    # X_train, X_eval, y_train, y_eval = train_test_split(X_dev, y_dev, test_size=0.3, random_state=42, shuffle=True)

    sc = StandardScaler()
    X_train, X_eval = sc.transform(X_train), sc.transform(X_eval)

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[-1])
    X_eval = X_eval.reshape(X_eval.shape[0], 1, X_eval.shape[-1])

    y_train -= 1
    y_eval -= 1

    # todo for future, transform them in one-hot encoded versions
    # y_train = y_train.reshape(y_train.shape[0], 1, 1)
    # y_eval = y_eval.reshape(y_eval.shape[0], 1, 1)

    train_dataset, eval_dataset = ArrayDataset(X_train, y_train), ArrayDataset(X_eval, y_eval)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=epoch_shuffle)
    eval_dataloader = DataLoader(eval_dataset, batch_size=len(eval_dataset))

    # Optimizer and loss
    optimizer = cm.SGD(lr=1e-5, momentum=0.9)
    loss = cm.CrossEntropyLoss(clip_value=1e-7, reduction='mean')
    # loss = cm.MSELoss(const=0.5, reduction='mean')

    model = get_model_classification(54, hidden_sizes, 7, winit_low, winit_high)  # , l2_lambda=1e-7)
    # get_model_regression(10, hidden_sizes, 1, winit_low, winit_high, l2_lambda=1e-7)
    model.compile(
        optimizer, loss, metrics=[SparseCategoricalAccuracy()]
    )
    optim_state = []
    early_stopping = EarlyStopping(
        monitor='Val_SparseCategoricalAccuracy', min_delta=1e-3, patience=20, mode='max', return_best_result=True
    )
    history = model.train(
        train_dataloader, eval_dataloader, max_epochs=100, callbacks=[
            TrainingCSVLogger(float_round_val=8), OptimizerMonitor(optim_state), early_stopping, InteractiveLogger(),
            InteractiveLogger(),
        ]
    )
    print(f'Training stopped at epoch {len(history)}, '
          f'best model reached at {early_stopping.get_best_epoch()} '
          f'with a value of {early_stopping.get_best_value()} '
          f'(last reference value is = {early_stopping.last_value_recorded} with min delta = {early_stopping.min_delta})')
    plot_history(0, history=history)


if __name__ == '__main__':
    # test_iris(hidden_sizes=(2,))
    test_breast_cancer(hidden_sizes=(4,))
    # test_covtype(hidden_sizes=(16,))
    exit(0)
