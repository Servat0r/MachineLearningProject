# Comparison tests on scikit-learn datasets with tests/scikit_learn_datasets_regression.py
from __future__ import annotations
from sklearn.datasets import load_breast_cancer, fetch_covtype  # Classification datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tests import *
import tensorflow as tf


def keras_test_breast_cancer(hidden_sizes=(4,), winit_low=-0.1, winit_high=0.1, epoch_shuffle=True):

    # Load and split dataset
    X, y = load_breast_cancer(return_X_y=True)
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
    # X_train, X_eval, y_train, y_eval = train_test_split(X_dev, y_dev, test_size=0.3, random_state=42, shuffle=True)

    X_train, X_eval, y_train, y_eval = X_dev, X_test, y_dev, y_test

    sc = StandardScaler()
    X_train, X_eval = sc.fit_transform(X_train), sc.transform(X_eval)

    # X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[-1])
    # X_eval = X_eval.reshape(X_eval.shape[0], 1, X_eval.shape[-1])

    # y_train = y_train.reshape(y_train.shape[0], 1, 1)
    # y_eval = y_eval.reshape(y_eval.shape[0], 1, 1)

    # Optimizer and loss
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)  # , momentum=0.9)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    # loss = tf.keras.losses.MeanSquaredError()

    model = tf.keras.Sequential()
    # initializer = tf.keras.initializers.RandomUniform(winit_low, winit_high, seed=0)
    l2_regularizer = tf.keras.regularizers.L2(1e-5)
    # l2_regularizer = None
    for hsize in hidden_sizes:
        model.add(
            tf.keras.layers.Dense(
                hsize, activation='tanh',
                kernel_initializer=tf.keras.initializers.RandomUniform(winit_low, winit_high),
                kernel_regularizer=l2_regularizer
            )
        )
    model.add(tf.keras.layers.Dense(
        2, activation='linear',
        kernel_initializer=tf.keras.initializers.RandomUniform(winit_low, winit_high),
        kernel_regularizer=l2_regularizer
    ))
    model.add(tf.keras.layers.Softmax())
    model.compile(
        optimizer, loss, metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='sca')],
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_sca', min_delta=1e-6, patience=100, mode='max')
    history = model.fit(
        X_train, y_train, batch_size=4, validation_data=(X_eval, y_eval), epochs=1000, callbacks=[early_stopping],
        shuffle=epoch_shuffle,
    )
    print(f'Training stopped at epoch {early_stopping.stopped_epoch}, best model reached at {early_stopping.best_epoch}')
    keras_plot_history(0, history=history, n_epochs=early_stopping.stopped_epoch)


def keras_test_covtype(hidden_sizes, winit_low=-0.1, winit_high=0.1, epoch_shuffle=True):

    # Load and split dataset
    X, y = fetch_covtype(return_X_y=True, shuffle=True)
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
    X_train, X_eval, y_train, y_eval = X_dev, X_test, y_dev, y_test
    # X_train, X_eval, y_train, y_eval = train_test_split(X_dev, y_dev, test_size=0.3, random_state=42, shuffle=True)

    sc = StandardScaler()
    X_train, X_eval = sc.fit_transform(X_train), sc.transform(X_eval)

    # X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[-1])
    # X_eval = X_eval.reshape(X_eval.shape[0], 1, X_eval.shape[-1])

    y_train -= 1
    y_eval -= 1

    # Optimizer and loss
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    model = tf.keras.Sequential()
    l2_regularizer = tf.keras.regularizers.L2(1e-5)
    for hsize in hidden_sizes:
        model.add(
            tf.keras.layers.Dense(
                hsize, activation='tanh',
                kernel_initializer=tf.keras.initializers.RandomUniform(winit_low, winit_high),
                kernel_regularizer=l2_regularizer
            )
        )
    model.add(tf.keras.layers.Dense(
        7, activation='linear',
        kernel_initializer=tf.keras.initializers.RandomUniform(winit_low, winit_high),
        kernel_regularizer=l2_regularizer
    ))
    model.add(tf.keras.layers.Softmax())
    model.compile(
        optimizer, loss, metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='sca')],
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_sca', min_delta=1e-3, patience=20, mode='max')
    history = model.fit(
        X_train, y_train, batch_size=32, validation_data=(X_eval, y_eval), epochs=100, callbacks=[early_stopping],
        shuffle=epoch_shuffle,
    )
    print(f'Training stopped at epoch {early_stopping.stopped_epoch}, best model reached at {early_stopping.best_epoch}')
    keras_plot_history(0, history=history, n_epochs=early_stopping.stopped_epoch)


if __name__ == '__main__':
    # keras_test_breast_cancer(hidden_sizes=(4,))
    keras_test_covtype(hidden_sizes=(16,))
    exit(0)
