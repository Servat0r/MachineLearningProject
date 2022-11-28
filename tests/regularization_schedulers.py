from __future__ import annotations
from tests.utils import *
import core.utils as cu
from core.data import DataLoader
import core.modules as cm


def __generate_layers(weigths_regularizer, biases_regularizer, low=-1.0, high=1.0):
    dense1 = cm.Dense(
            INPUT_DIM, HIDDEN_SIZE, cm.Tanh(),
            weights_initializer=cu.RandomUniformInitializer(low, high), grad_reduction='mean',
            # biases_initializer=cu.RandomUniformInitializer(-1.0, 1.0),
            weights_regularizer=weigths_regularizer, biases_regularizer=biases_regularizer,
        )
    dense2 = cm.Dense(
        HIDDEN_SIZE, HIDDEN_SIZE, cm.Tanh(),
        weights_initializer=cu.RandomUniformInitializer(low, high), grad_reduction='mean',
        # biases_initializer=cu.RandomUniformInitializer(-1.0, 1.0),
        weights_regularizer=weigths_regularizer, biases_regularizer=biases_regularizer,
    )
    linear3 = cm.Linear(
        HIDDEN_SIZE, OUTPUT_DIM,
        weights_initializer=cu.RandomUniformInitializer(low, high), grad_reduction='mean',
        # biases_initializer=cu.RandomUniformInitializer(-1.0, 1.0),
        weights_regularizer=weigths_regularizer, biases_regularizer=biases_regularizer,
    )
    return [dense1, dense2, linear3]


def test_fully_connected_minibatch_model_with_regularizations(
        n_epochs=5000, mb_size=1, epoch_shuffle=True, func=arange_square_data,
        l1_regularizer=0., l2_regularizer=0., lr=0.001, start_plot_epoch=0, func_args: dict = None,
):
    # Generate train dataset
    x, y, train_dataset, accuracy_precision = generate_dataset(func)

    # Generate validation dataset
    func_args = {} if func_args is None else func_args
    x_eval, y_eval = func(samples=N_SAMPLES//5, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, **func_args)
    eval_dataset = ArrayDataset(x_eval, y_eval)

    # Generate dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=mb_size, shuffle=epoch_shuffle)
    eval_dataloader = DataLoader(eval_dataset, batch_size=N_SAMPLES//5)
    model = cm.Model(
        __generate_layers(
            weigths_regularizer=cm.L1L2Regularizer(l1_lambda=l1_regularizer, l2_lambda=l2_regularizer),
            biases_regularizer=cm.L1L2Regularizer(l1_lambda=l1_regularizer, l2_lambda=l2_regularizer),
        )
    )
    optimizer = cm.SGD(lr=lr)
    loss_function = cm.MSELoss(reduction='mean')
    # Use Model class for training and epoch losses recording
    model.compile(optimizer=optimizer, loss=loss_function)
    train_epoch_losses, eval_epoch_losses, optimizer_state = np.empty(n_epochs), np.empty(n_epochs), []
    model.train(
        train_dataloader, eval_dataloader, n_epochs=n_epochs, train_epoch_losses=train_epoch_losses,
        eval_epoch_losses=eval_epoch_losses, optimizer_state=optimizer_state,
    )
    for epoch, (epoch_tr_loss, epoch_ev_loss, optim_state) in \
            enumerate(zip(train_epoch_losses, eval_epoch_losses, optimizer_state)):
        print(f'epoch: {epoch} ' +
              f'(tr_loss: {epoch_tr_loss.item():.8f}, ' +
              f'ev_loss: {epoch_ev_loss.item():.8f}), ' +
              f'optim_state: {optim_state}')

    plot_losses(start_plot_epoch, train_epoch_losses, eval_epoch_losses)


def test_fc_minibatch_model_with_regularizations_lrscheduler(
        n_epochs=5000, mb_size=1, epoch_shuffle=True, func=arange_square_data, lr=0.001, momentum=0.,
        lr_scheduler: cm.Scheduler = None, l1_regularizer=0., l2_regularizer=0., start_plot_epoch=0,
        func_args: dict = None,
):
    # Generate train dataset
    x, y, train_dataset, accuracy_precision = generate_dataset(func)

    # Generate validation dataset
    func_args = {} if func_args is None else func_args
    x_eval, y_eval = func(samples=N_SAMPLES//5, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, **func_args)
    eval_dataset = ArrayDataset(x_eval, y_eval)

    # Generate dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=mb_size, shuffle=epoch_shuffle)
    eval_dataloader = DataLoader(eval_dataset, batch_size=N_SAMPLES//5)
    model = cm.Model(
        __generate_layers(
            weigths_regularizer=cm.L1L2Regularizer(l1_lambda=l1_regularizer, l2_lambda=l2_regularizer),
            biases_regularizer=cm.L1L2Regularizer(l1_lambda=l1_regularizer, l2_lambda=l2_regularizer),
        )
    )
    # Shadow optimizer in order to use custom lr and scheduler
    loss_function = cm.MSELoss(reduction='mean')
    optimizer = cm.SGD(lr=lr, lr_decay_scheduler=lr_scheduler, momentum=momentum)

    # Use Model class for training and epoch losses recording
    model.compile(optimizer=optimizer, loss=loss_function)
    train_epoch_losses, eval_epoch_losses, optimizer_state = np.empty(n_epochs), np.empty(n_epochs), []
    model.train(
        train_dataloader, eval_dataloader, n_epochs=n_epochs, train_epoch_losses=train_epoch_losses,
        eval_epoch_losses=eval_epoch_losses, optimizer_state=optimizer_state,
    )

    # Print results
    for epoch, (epoch_tr_loss, epoch_ev_loss, optim_state) in \
            enumerate(zip(train_epoch_losses, eval_epoch_losses, optimizer_state)):
        print(f'epoch: {epoch} ' +
              f'(tr_loss: {epoch_tr_loss.item():.8f}, ' +
              f'ev_loss: {epoch_ev_loss.item():.8f}), ' +
              f'optim_state: {optim_state}')

    plot_losses(start_plot_epoch, train_epoch_losses, eval_epoch_losses)
