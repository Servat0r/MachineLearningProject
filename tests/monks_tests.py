# Tests with MONKs datasets
from __future__ import annotations
from tests.utils import *
from core.utils.types import *
from core.data import *
from core.callbacks import InteractiveLogger, TrainingCSVLogger, EarlyStopping
from core.metrics import BinaryAccuracy
import core.utils as cu
import core.modules as cm


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
    train_data, train_labels, eval_data, eval_labels = read_monk(
        train_file, dirpath, shuffle_once=shuffle, shuffle_seed=0,
        validation_size=validation_size, dtype=dtype
    )
    test_data, test_labels, _, _ = read_monk(test_file, dirpath, validation_size=None)
    train_dataset = ArrayDataset(train_data, train_labels)
    eval_dataset = None if eval_data is None else ArrayDataset(eval_data, eval_labels)
    test_dataset = ArrayDataset(test_data, test_labels)
    layers = [cm.Input()]
    sizes = [in_size] + list(hidden_sizes)
    for i in range(len(sizes)-1):
        p, q = sizes[i], sizes[i+1]
        layers.append(
            cm.Dense(p, q, cm.Tanh(),
                     cu.RandomUniformInitializer(winit_low, winit_high), gradients_reduction=grad_reduction)
        )
    layers.append(
        cm.Dense(sizes[-1], out_size, cm.Sigmoid(),
                 cu.RandomUniformInitializer(winit_low, winit_high), gradients_reduction=grad_reduction)
    )
    model = cm.Model(layers)
    return model, train_dataset, eval_dataset, test_dataset


# todo add L2 lambda
def test_monk(
        model, train_dataset, eval_dataset, test_dataset, lr=1e-1, momentum=0., batch_size=1,
        n_epochs=100, metrics=None, callbacks=None, metrics_to_plot=None, ylabels=None,
        plot_save_paths=None, model_save_path=None,
):
    optimizer = cm.SGD(lr=lr, momentum=momentum)
    loss = cm.MSELoss(const=1., reduction='mean')

    # Configure optional parameters for training and plotting
    metrics = [BinaryAccuracy()] if metrics is None else metrics
    callbacks = [] if callbacks is None else callbacks
    callbacks.append(InteractiveLogger())
    if metrics_to_plot is None:
        metrics_to_plot = [
            {
                'loss': "Training",
                'Val_loss': "Test",
            },
            {
                'BinaryAccuracy': "Training",
                'Val_BinaryAccuracy': "Test",
            }
        ]
    plot_save_paths = ['../results/monks/monk1_losses.png', '../results/monks/monk1_accuracy.png'] \
        if plot_save_paths is None else plot_save_paths
    ylabels = ['Loss (MSE)', 'BinaryAccuracy'] if ylabels is None else ylabels

    # Configure train and validation dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if eval_dataset is not None:
        eval_dataloader = DataLoader(eval_dataset, batch_size=len(eval_dataset))
    else:
        eval_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))
    model.compile(optimizer, loss, metrics=metrics)

    # Training and results plotting
    history = model.train(train_dataloader, eval_dataloader, max_epochs=n_epochs, callbacks=callbacks)
    model.set_to_eval()
    if model_save_path is not None:
        dirname = os.path.dirname(model_save_path)
        os.makedirs(dirname, exist_ok=True)
        model.save(model_save_path, include_compile_objs=True, include_history=True)
    for metric, save_path, ylabel in zip(metrics_to_plot, plot_save_paths, ylabels):
        plot_metrics(history, metric, save_path, len(history), ylabel=ylabel, makedirs=True)
    return history


def test_monk1(
        validation_size=None, lr=1e-1, momentum=0., reduction='mean', batch_size=1, shuffle=True,
        n_epochs=100, metrics=None, callbacks=None, metrics_to_plot=None, ylabels=None,
        plot_save_paths=None, model_save_path=None, dir_path='../datasets/monks',
        csv_save_path=None,
):
    model, train_dataset, eval_dataset, test_dataset = get_monk_setup_hold_out(
        train_file=MONK1_TRAIN, test_file=MONK1_TEST, hidden_sizes=MONK1_HIDDEN_SIZES,
        validation_size=validation_size, grad_reduction=reduction, shuffle=shuffle,
        dirpath=dir_path,
    )
    callbacks = [] if callbacks is None else callbacks
    callbacks.append(TrainingCSVLogger(csv_save_path, 'monk1_log.csv'))
    return test_monk(
        model, train_dataset, eval_dataset, test_dataset, lr, momentum, batch_size, n_epochs,
        metrics, callbacks, metrics_to_plot, ylabels, plot_save_paths, model_save_path,
    )


def test_monk2(
        validation_size=None, lr=1e-1, momentum=0., reduction='mean', batch_size=1, shuffle=True,
        n_epochs=100, metrics=None, callbacks=None, metrics_to_plot=None, ylabels=None,
        plot_save_paths=None, model_save_path=None, dir_path='../datasets/monks',
        csv_save_path=None,
):
    model, train_dataset, eval_dataset, test_dataset = get_monk_setup_hold_out(
        train_file=MONK2_TRAIN, test_file=MONK2_TEST, hidden_sizes=MONK2_HIDDEN_SIZES,
        validation_size=validation_size, grad_reduction=reduction, shuffle=shuffle,
        dirpath=dir_path,
    )
    callbacks = [] if callbacks is None else callbacks
    callbacks.append(TrainingCSVLogger(csv_save_path, 'monk2_log.csv'))
    return test_monk(
        model, train_dataset, eval_dataset, test_dataset, lr, momentum, batch_size, n_epochs,
        metrics, callbacks, metrics_to_plot, ylabels, plot_save_paths, model_save_path,
    )


def test_monk3(
        validation_size=None, lr=1e-1, momentum=0., reduction='mean', batch_size=1, shuffle=True,
        n_epochs=100, metrics=None, callbacks=None, metrics_to_plot=None, ylabels=None,
        plot_save_paths=None, model_save_path=None, dir_path='../datasets/monks',
        csv_save_path=None,
):
    model, train_dataset, eval_dataset, test_dataset = get_monk_setup_hold_out(
        train_file=MONK3_TRAIN, test_file=MONK3_TEST, hidden_sizes=MONK3_HIDDEN_SIZES,
        validation_size=validation_size, grad_reduction=reduction, shuffle=shuffle,
        dirpath=dir_path,
    )
    callbacks = [] if callbacks is None else callbacks
    callbacks.append(TrainingCSVLogger(csv_save_path, 'monk3_log.csv'))
    callbacks.append(EarlyStopping(monitor='Val_loss', min_delta=1e-2, patience=10))
    return test_monk(
        model, train_dataset, eval_dataset, test_dataset, lr, momentum, batch_size, n_epochs,
        metrics, callbacks, metrics_to_plot, ylabels, plot_save_paths, model_save_path,
    )


if __name__ == '__main__':
    test_monk1(
        lr=0.2, batch_size=1, shuffle=True, n_epochs=100, model_save_path='../results/monks/monk1_model.model',
        plot_save_paths=['../results/monks/monk1_losses.png', '../results/monks/monk1_accuracy.png'],
        csv_save_path='../results/monks',
    )
    test_monk2(
        lr=1e-1, batch_size=1, shuffle=True, n_epochs=200, model_save_path='../results/monks/monk2_model.model',
        plot_save_paths=['../results/monks/monk2_losses.png', '../results/monks/monk2_accuracy.png'],
        csv_save_path='../results/monks',
    )
    test_monk3(
        lr=1e-2, batch_size=2, shuffle=True, n_epochs=200, model_save_path='../results/monks/monk3_model.model',
        plot_save_paths=['../results/monks/monk3_losses.png', '../results/monks/monk3_accuracy.png'],
        csv_save_path='../results/monks',
    )


__all__ = ['get_monk_setup_hold_out', 'test_monk', 'test_monk1', 'test_monk2', 'test_monk3']
