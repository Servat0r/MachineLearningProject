from __future__ import annotations
from tests.utils import *
import core.modules as cm
from core.callbacks import OptimizerMonitor


def test_fully_connected_minibatch_model_with_regularizations(
        n_epochs=5000, mb_size=1, epoch_shuffle=True, func=arange_square_data,
        l1_regularizer=0., l2_regularizer=0., lr=0.001, start_plot_epoch=0, func_args: dict = None,
):
    train_dataloader, eval_dataloader, model = generate_dataset_and_model(
        func, func_args, N_SAMPLES//5, mb_size, epoch_shuffle, winit_low=-0.7, winit_high=0.7,
        l1_lambda=l1_regularizer, l2_lambda=l2_regularizer,
    )
    optimizer = cm.SGD(lr=lr)
    loss_function = cm.MSELoss(reduction='mean')
    # Use Model class for training and epoch losses recording
    model.compile(optimizer=optimizer, loss=loss_function)
    optimizer_state = []
    optim_monitor = OptimizerMonitor(optimizer_state)
    history = model.train(
        train_dataloader, eval_dataloader, n_epochs=n_epochs, callbacks=[optim_monitor],
    )
    train_epoch_losses, eval_epoch_losses = history['loss'], history['Val_loss']
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
    train_dataloader, eval_dataloader, model = generate_dataset_and_model(
        func, func_args, N_SAMPLES//5, mb_size, epoch_shuffle, winit_low=-0.7,
        winit_high=0.7, l1_lambda=l1_regularizer, l2_lambda=l2_regularizer,
    )
    loss_function = cm.MSELoss(reduction='mean')
    optimizer = cm.SGD(lr=lr, lr_decay_scheduler=lr_scheduler, momentum=momentum)

    # Use Model class for training and epoch losses recording
    model.compile(optimizer=optimizer, loss=loss_function)
    optimizer_state = []
    optim_monitor = OptimizerMonitor(optimizer_state)
    history = model.train(
        train_dataloader, eval_dataloader, n_epochs=n_epochs, callbacks=[optim_monitor],
    )
    train_epoch_losses, eval_epoch_losses = history['loss'], history['Val_loss']
    # Print results
    for epoch, (epoch_tr_loss, epoch_ev_loss, optim_state) in \
            enumerate(zip(train_epoch_losses, eval_epoch_losses, optimizer_state)):
        print(f'epoch: {epoch} ' +
              f'(tr_loss: {epoch_tr_loss.item():.8f}, ' +
              f'ev_loss: {epoch_ev_loss.item():.8f}), ' +
              f'optim_state: {optim_state}')
    plot_losses(start_plot_epoch, train_epoch_losses, eval_epoch_losses)
