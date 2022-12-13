# MLP with MONKs datasets
from __future__ import annotations
from tests.utils import *
from core.utils.types import *
from core.data import *
from core.callbacks import InteractiveLogger, TrainingCSVLogger
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
MONK_INSIZE = 17
MONK_HIDDEN_SIZES = (4,)
MONK_OUTSIZE = 1



# Setup functions for each of the MONKs datasets
def mlp_monk_neural_network(
        dirpath: str = '../datasets/monks', train_file=MONK1_TRAIN, test_file=MONK1_TEST,
        in_size=MONK_INSIZE, hidden_sizes=MONK_HIDDEN_SIZES, out_size=MONK_OUTSIZE,
        validation_size=None, grad_reduction='mean', shuffle=True,
        winit_low=-0.1, winit_high=0.1, dtype=np.float64, lr=1e-1, momentum=0., batch_size=1,
        n_epochs=100, model_save_path=None,
):
    # Get datasets
    train_data, train_labels, eval_data, eval_labels = read_monk(train_file, dirpath, shuffle, validation_size, dtype)
    test_data, test_labels, _, _ = read_monk(test_file, dirpath, validation_size=None)
    train_dataset = ArrayDataset(train_data, train_labels)
    eval_dataset = None if eval_data is None else ArrayDataset(eval_data, eval_labels)
    test_dataset = ArrayDataset(test_data, test_labels)

    # Defining model
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

    # define optimizer to use and loss funtion
    optimizer = cm.SGD(lr=lr, momentum=momentum)
    loss = cm.MSELoss(const=1., reduction='mean')

    # Configure optional parameters for training and plotting
    metrics = [BinaryAccuracy()] if metrics is None else metrics
    callbacks = [InteractiveLogger(), TrainingCSVLogger()] if callbacks is None else callbacks

    metrics_to_plot = [['loss', 'Val_loss'], ['BinaryAccuracy', 'Val_BinaryAccuracy']] \
        if metrics_to_plot is None else metrics_to_plot
    plot_save_paths = ['../results/monks/monk_losses.png', '../results/monks/monk_accuracy.png'] \
        if plot_save_paths is None else plot_save_paths
    ylabels = ['Loss', 'Accuracy'] if ylabels is None else ylabels

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
        plot_metrics(history, metric, save_path, n_epochs, ylabel=ylabel, makedirs=True)

    pass
