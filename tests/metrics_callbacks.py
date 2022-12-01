# Tests for metrics and callbacks
from __future__ import annotations
from tests.utils import *
from core.utils.types import *
from core.data import DataLoader
import core.modules as cm
import core.metrics as cmt
import core.callbacks as cc


def test_fully_connected_minibatch_regularization_metrics(
        n_epochs=5000, mb_size=1, epoch_shuffle=True, func=arange_square_data,
        l1_regularizer=0., l2_regularizer=0., lr=0.001, start_plot_epoch=0, func_args: dict = None,
        metrics: cmt.Metric | Sequence[cmt.Metric] = None,
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
        generate_layers(
            low=-0.7, high=0.7,
            weights_regularizer=cm.L1L2Regularizer(l1_lambda=l1_regularizer, l2_lambda=l2_regularizer),
            biases_regularizer=cm.L1L2Regularizer(l1_lambda=l1_regularizer, l2_lambda=l2_regularizer),
        )
    )
    optimizer = cm.SGD(lr=lr)
    loss_function = cm.MSELoss(reduction='mean')
    # Use Model class for training and epoch losses recording
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
    optimizer_state = []
    optim_monitor = cc.OptimizerMonitor(optimizer_state)
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

    history_losses = history['loss']
    assert np.equal(history_losses, train_epoch_losses).all()
    plot_history(start_plot_epoch, history=history)


def test_fully_connected_regularization_metrics_logging(
    n_epochs=5000, mb_size=1, epoch_shuffle=True, func=arange_square_data,
    l1_regularizer=0., l2_regularizer=0., lr=0.001, start_plot_epoch=0, func_args: dict = None,
    metrics: cmt.Metric | Sequence[cmt.Metric] = None, train_log_file='train_log.csv', round_val=None,
    test_log_file='test_log.csv', include_mb=False,
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
        generate_layers(
            low=-0.7, high=0.7,
            weights_regularizer=cm.L1L2Regularizer(l1_lambda=l1_regularizer, l2_lambda=l2_regularizer),
            biases_regularizer=cm.L1L2Regularizer(l1_lambda=l1_regularizer, l2_lambda=l2_regularizer),
        )
    )
    optimizer = cm.SGD(lr=lr)
    loss_function = cm.MSELoss(reduction='mean')
    # Use Model class for training and epoch losses recording
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
    optimizer_state = []
    optim_monitor = cc.OptimizerMonitor(optimizer_state)
    history = model.train(
        train_dataloader, eval_dataloader, n_epochs=n_epochs, callbacks=[
            cc.TrainingCSVLogger(train_fpath=train_log_file, round_val=round_val, include_mb=include_mb),
            optim_monitor,
        ],
    )
    train_epoch_losses, eval_epoch_losses = history['loss'], history['Val_loss']
    for epoch, (epoch_tr_loss, epoch_ev_loss, optim_state) in \
            enumerate(zip(train_epoch_losses, eval_epoch_losses, optimizer_state)):
        print(f'epoch: {epoch} ' +
              f'(tr_loss: {epoch_tr_loss.item():.8f}, ' +
              f'ev_loss: {epoch_ev_loss.item():.8f}), ' +
              f'optim_state: {optim_state}')

    history_losses = history['loss']
    assert np.equal(history_losses, train_epoch_losses).all()
    plot_history(start_plot_epoch, eval_epoch_losses, history=history)

    # Now test also TestCSVLogger  todo finish!
    model.set_to_test()
    if 'start' in func_args.keys():
        func_args['start'] += N_SAMPLES//5 * INPUT_DIM  # offset to avoid using validation data for testing
    x_test, y_test = func(samples=10, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, **func_args)
    y_hat_test = model.forward(x_test)
    print('To be finished!')
