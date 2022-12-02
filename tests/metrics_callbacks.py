# Tests for metrics and callbacks
from __future__ import annotations
from tests.utils import *
from core.utils.types import *
import core.modules as cm
import core.metrics as cmt
import core.callbacks as cc


def test_fully_connected_minibatch_regularization_metrics(
        n_epochs=5000, mb_size=1, epoch_shuffle=True, func=arange_square_data,
        l1_regularizer=0., l2_regularizer=0., lr=0.001, start_plot_epoch=0, func_args: dict = None,
        metrics: cmt.Metric | Sequence[cmt.Metric] = None,
):
    train_dataloader, eval_dataloader, model = generate_dataset_and_model(
        func, func_args, N_SAMPLES//5, mb_size, epoch_shuffle, winit_low=-0.7,
        winit_high=0.7, l1_lambda=l1_regularizer, l2_lambda=l2_regularizer,
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
    train_dataloader, eval_dataloader, model = generate_dataset_and_model(
        func, func_args, N_SAMPLES//5, mb_size, epoch_shuffle, winit_low=-0.7,
        winit_high=0.7, l1_lambda=l1_regularizer, l2_lambda=l2_regularizer,
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
    plot_history(start_plot_epoch, history=history)

    # Now test also TestCSVLogger  todo finish!
    model.set_to_test()
    if 'start' in func_args.keys():
        func_args['start'] += N_SAMPLES//5 * INPUT_DIM  # offset to avoid using validation data for testing
    x_test, y_test = func(samples=10, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, **func_args)
    y_hat_test = model.forward(x_test)
    print('To be finished!')


def test_model_checkpoint_and_backup(
        n_epochs=5000, mb_size=1, epoch_shuffle=True, func=arange_square_data,
        l1_regularizer=0., l2_regularizer=0., lr=0.001, start_plot_epoch=0, func_args: dict = None,
        metrics: cmt.Metric | Sequence[cmt.Metric] = None, train_log_file='train_log.csv', round_val=None,
        model_checkpoint_fpath='model_checkpoint.model', model_backup_fpath='model_backup.model',
        save_every=10,
):
    # Test for ModelCheckpoint and ModelBackup callbacks
    train_dataloader, eval_dataloader, model = generate_dataset_and_model(
        func, func_args, N_SAMPLES//5, mb_size, epoch_shuffle, winit_low=-0.7,
        winit_high=0.7, l1_lambda=l1_regularizer, l2_lambda=l2_regularizer,
    )
    optimizer = cm.SGD(lr=lr)
    loss_function = cm.MSELoss(reduction='mean')
    # Use Model class for training and epoch losses recording
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
    history = model.train(
        train_dataloader, eval_dataloader, n_epochs=n_epochs, callbacks=[
            cc.TrainingCSVLogger(train_fpath=train_log_file, round_val=round_val, include_mb=True),
            cc.ModelCheckpoint(fpath=model_checkpoint_fpath, save_every=save_every),
            cc.ModelBackup(fpath=model_backup_fpath, save_every=save_every),
            WaitKey(wait_every=save_every, prompt='Press any key to continue ...'),
        ],
    )

    model2 = cm.Model.load(model_checkpoint_fpath)
    assert model.equal(model2)
    print('ModelCheckpoint test passed!')

    model3 = cm.Model.load(model_backup_fpath)
    assert model == model3
    print('ModelBackup test passed!')


def test_early_stopping(
        n_epochs=5000, mb_size=1, epoch_shuffle=True, func=arange_square_data,
        l1_regularizer=0., l2_regularizer=0., lr=0.001, start_plot_epoch=0, func_args: dict = None,
        metrics: cmt.Metric | Sequence[cmt.Metric] = None, train_log_file='train_log.csv', round_val=None,
        min_delta=1e-2, mode='min', monitor='Val_loss', patience=10,
):
    # Test for ModelCheckpoint and ModelBackup callbacks
    train_dataloader, eval_dataloader, model = generate_dataset_and_model(
        func, func_args, N_SAMPLES//5, mb_size, epoch_shuffle, winit_low=-0.7,
        winit_high=0.7, l1_lambda=l1_regularizer, l2_lambda=l2_regularizer,
    )
    optimizer = cm.SGD(lr=lr)
    loss_function = cm.MSELoss(reduction='mean')
    # Use Model class for training and epoch losses recording
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
    early_stopping = cc.EarlyStopping(monitor, min_delta, patience, mode, return_best=True)
    history = model.train(
        train_dataloader, eval_dataloader, n_epochs=n_epochs, callbacks=[
            cc.TrainingCSVLogger(train_fpath=train_log_file, round_val=round_val, include_mb=True),
            early_stopping,
        ],
    )
    best_epoch = early_stopping.get_best_epoch()
    print(f'Maximum number of epochs was: {n_epochs}, '
          f'while effective ones are {len(history)} and best model was obtained in epoch {best_epoch}')
    print(
        f"Monitor metric for best model is {history[monitor][best_epoch]}, "
        f"while for last one is {history[monitor][len(history)-1]}"
    )
    # last, best = model, early_stopping.get_best()
    # eval_dataloader.before_epoch()
    # X_eval, y_eval = next(eval_dataloader)

    plot_history(start_plot_epoch, history=history)
