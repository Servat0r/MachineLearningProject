from __future__ import annotations
from tests.utils import *
import core.modules as cm
from core.callbacks import OptimizerMonitor


def test_separated(func=arange_square_data, lr=0.001):
    linear1, activation1, linear2, activation2, linear3 = generate_layers()
    loss_function = cm.MSELoss(reduction='mean')
    optimizer = cm.SGD(lr=lr)

    x, y, train_dataset = generate_dataset(func)
    for epoch in range(5001):
        optimizer.before_epoch()

        # Perform a forward pass of our training data through this layer
        linear1.forward(x)

        # Perform a forward pass through activation function
        # takes the output of first dense layer here
        activation1.forward(linear1.output)

        # Perform a forward pass through second Dense layer
        # takes outputs of activation function
        # of first layer as inputs
        linear2.forward(activation1.output)

        # Perform a forward pass through activation function
        # takes the output of second dense layer here
        activation2.forward(linear2.output)

        # Perform a forward pass through third Dense layer
        # takes outputs of activation function of second layer as inputs
        linear3.forward(activation2.output)

        data_loss = loss_function(linear3.output, y)

        loss = data_loss
        predictions = linear3.output

        if not epoch % 10:
            print(f'epoch: {epoch}, ' +
                  f'loss: {loss.item():.4f} (' +
                  f'data_loss: {data_loss.item():.4f}, ' +
                  # f'reg_loss: {regularization_loss:.8f}), ' +
                  f'lr: {optimizer.current_learning_rate}')

        # Backward pass
        dvals = loss_function.backward(linear3.output, y)
        dvals = linear3.backward(dvals)
        dvals = activation2.backward(dvals)
        dvals = linear2.backward(dvals)
        dvals = activation1.backward(dvals)
        linear1.backward(dvals)

        # Update weights and biases
        optimizer.update([linear1, linear2, linear3])
        optimizer.after_epoch()


def test_fully_connected_minibatch_model(
        n_epochs=5000, mb_size=1, epoch_shuffle=True, func=arange_square_data, lr=0.001, func_args: dict = None,
):
    loss_function = cm.MSELoss(reduction='mean')
    optimizer = cm.SGD(lr=lr)
    train_dataloader, eval_dataloader, model = generate_dataset_and_model(
        func, func_args, N_SAMPLES//5, mb_size, epoch_shuffle, winit_low=-0.7,
        winit_high=0.7, l1_lambda=0., l2_lambda=0.,
    )
    # Use Model class for training and epoch losses recording
    model.compile(optimizer=optimizer, loss=loss_function)
    optimizer_state = []
    optim_monitor = OptimizerMonitor(optimizer_state)
    history = model.train(
        train_dataloader, eval_dataloader, max_epochs=n_epochs, callbacks=[optim_monitor],
    )
    train_epoch_losses, eval_epoch_losses = history['loss'], history['Val_loss']
    for epoch, (epoch_tr_loss, epoch_ev_loss, optim_state) in \
            enumerate(zip(train_epoch_losses, eval_epoch_losses, optimizer_state)):
        print(f'epoch: {epoch} ' +
              f'(tr_loss: {epoch_tr_loss.item():.8f}, ' +
              f'ev_loss: {epoch_ev_loss.item():.8f}), ' +
              f'optim_state: {optim_state}')

    plot_losses(0, train_epoch_losses, eval_epoch_losses)
