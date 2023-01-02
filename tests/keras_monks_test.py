# Tests with MONKs datasets
from __future__ import annotations
from tests.utils import *
from core.utils.types import *
from core.data import *

import tensorflow as tf
layers = tf.keras.layers


# MONKs filenames
MONK1_TRAIN = 'monks-1.train'
MONK1_TEST = 'monks-1.test'
MONK2_TRAIN = 'monks-2.train'
MONK2_TEST = 'monks-2.test'
MONK3_TRAIN = 'monks-3.train'
MONK3_TEST = 'monks-3.test'

# Other MONKs constants
MONK1_INSIZE = 17
MONK1_HIDDEN_SIZES = (4,)
MONK1_OUTSIZE = 1

MONK2_INSIZE = 17
MONK2_HIDDEN_SIZES = (4,)
MONK2_OUTSIZE = 1

MONK3_INSIZE = 17
MONK3_HIDDEN_SIZES = (15,)
MONK3_OUTSIZE = 1


# Setup functions for each of the MONKs datasets
def get_monk_setup_hold_out(
        dirpath: str = '../datasets/monks', train_file=MONK1_TRAIN, test_file=MONK1_TEST,
        in_size=MONK1_INSIZE, hidden_sizes=MONK1_HIDDEN_SIZES, out_size=MONK1_OUTSIZE,
        validation_size=None, grad_reduction='mean', shuffle=True,
        winit_low=-0.1, winit_high=0.1, dtype=np.float32,
):
    # Get datasets
    train_data, train_labels, eval_data, eval_labels = read_monk(train_file, dirpath, shuffle, validation_size, dtype)
    test_data, test_labels, _, _ = read_monk(test_file, dirpath, validation_size=None)

    # Create model
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(1, in_size)))
    sizes = [in_size] + list(hidden_sizes)
    for i in range(len(sizes)-1):
        p, q = sizes[i], sizes[i+1]
        model.add(
            layers.Dense(
                q, activation='tanh',
                kernel_initializer=tf.keras.initializers.RandomUniform(winit_low, winit_high)
            )
        )
    model.add(
        layers.Dense(
            out_size, activation='sigmoid',
            kernel_initializer=tf.keras.initializers.RandomUniform(winit_low, winit_high)
        )
    )
    return model, train_data, train_labels, eval_data, eval_labels, test_data, test_labels


def test_monk(
        model: tf.keras.Sequential, train_data, train_targets, eval_data, eval_targets, test_data, test_targets,
        lr=1e-1, momentum=0., batch_size=1, n_epochs=100, metrics=None, callbacks=None,
        metrics_to_plot=None, ylabels=None, plot_save_paths=None, model_save_path=None,
):
    optimizer = tf.keras.optimizers.SGD(lr, momentum)
    loss = tf.keras.losses.MeanSquaredError()
    metrics = [tf.keras.metrics.BinaryAccuracy()] if metrics is None else metrics
    callbacks = [] if callbacks is None else callbacks

    metrics_to_plot = [['loss', 'Val_loss'], ['BinaryAccuracy', 'Val_BinaryAccuracy']] \
        if metrics_to_plot is None else metrics_to_plot
    ylabels = ['Loss', 'Accuracy'] if ylabels is None else ylabels
    model.compile(optimizer, loss, metrics)

    history = model.fit(train_data, train_targets, batch_size=batch_size, epochs=n_epochs,
                        shuffle=True, callbacks=callbacks)
    # validation_data=(eval_data, eval_targets),
    keras_plot_history(0, history, n_epochs)
    return history


def test_monk1(
        validation_size=None, lr=1e-1, momentum=0., reduction='mean', batch_size=1, shuffle=True,
        n_epochs=100, metrics=None, callbacks=None, metrics_to_plot=None, ylabels=None,
        plot_save_paths=None, model_save_path=None,
):
    model, train_data, train_targets, eval_data, eval_targets, test_data, test_targets = get_monk_setup_hold_out(
        train_file=MONK1_TRAIN, test_file=MONK1_TEST, hidden_sizes=MONK1_HIDDEN_SIZES,
        validation_size=validation_size, grad_reduction=reduction, shuffle=shuffle,
    )
    return test_monk(
        model, train_data, train_targets, eval_data, eval_targets, test_data, test_targets,
        lr, momentum, batch_size, n_epochs, metrics, callbacks, metrics_to_plot, ylabels,
        plot_save_paths, model_save_path,
    )


def test_monk2(
        validation_size=None, lr=1e-1, momentum=0., reduction='mean', batch_size=1, shuffle=True,
        n_epochs=100, metrics=None, callbacks=None, metrics_to_plot=None, ylabels=None,
        plot_save_paths=None, model_save_path=None,
):
    model, train_data, train_targets, eval_data, eval_targets, test_data, test_targets = get_monk_setup_hold_out(
        train_file=MONK2_TRAIN, test_file=MONK2_TEST, hidden_sizes=MONK2_HIDDEN_SIZES,
        validation_size=validation_size, grad_reduction=reduction, shuffle=shuffle,
    )
    return test_monk(
        model, train_data, train_targets, eval_data, eval_targets, test_data, test_targets,
        lr, momentum, batch_size, n_epochs, metrics, callbacks, metrics_to_plot, ylabels,
        plot_save_paths, model_save_path,
    )


def test_monk3(
        validation_size=None, lr=1e-1, momentum=0., reduction='mean', batch_size=1, shuffle=True,
        n_epochs=100, metrics=None, callbacks=None, metrics_to_plot=None, ylabels=None,
        plot_save_paths=None, model_save_path=None,
):
    model, train_data, train_targets, eval_data, eval_targets, test_data, test_targets = get_monk_setup_hold_out(
        train_file=MONK3_TRAIN, test_file=MONK3_TEST, hidden_sizes=MONK3_HIDDEN_SIZES,
        validation_size=validation_size, grad_reduction=reduction, shuffle=shuffle,
    )
    return test_monk(
        model, train_data, train_targets, eval_data, eval_targets, test_data, test_targets,
        lr, momentum, batch_size, n_epochs, metrics, callbacks, metrics_to_plot, ylabels,
        plot_save_paths, model_save_path,
    )


if __name__ == '__main__':
    test_monk1(
        lr=1e-1, batch_size=1, shuffle=True, n_epochs=200, model_save_path='../results/monks/monk1_model.model',
        plot_save_paths=['../results/monks/monk1_losses.png', '../results/monks/monk1_accuracy.png'],
    )
    test_monk2(
        lr=1e-1, batch_size=1, shuffle=True, n_epochs=200, model_save_path='../results/monks/monk2_model.model',
        plot_save_paths=['../results/monks/monk2_losses.png', '../results/monks/monk2_accuracy.png'],
    )
    test_monk3(
        lr=1e-2, batch_size=2, shuffle=True, n_epochs=200,
        model_save_path='../results/monks/monk3_regularization_model.model',
        plot_save_paths=['../results/monks/monk3_losses.png', '../results/monks/monk3_accuracy.png'],
    )


__all__ = ['get_monk_setup_hold_out', 'test_monk', 'test_monk1', 'test_monk2', 'test_monk3']
