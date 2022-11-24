from __future__ import annotations
import math
from .utils import *
import core.utils as cu
from core.data import DataLoader
import core.modules as cm


def __generate_layers(low=-1.0, high=1.0):
    dense1 = cm.Linear(
        INPUT_DIM, 64,
        weights_initializer=cu.RandomUniformInitializer(low, high), grad_reduction='mean',
        # biases_initializer=cu.RandomUniformInitializer(-0.2, 0.2),
    )
    activation1 = cm.Tanh()
    dense2 = cm.Linear(
        64, 64, weights_initializer=cu.RandomUniformInitializer(low, high), grad_reduction='mean',
        # biases_initializer=cu.RandomUniformInitializer(-0.2, 0.2),
    )
    activation2 = cm.Tanh()
    dense3 = cm.Linear(
        64, OUTPUT_DIM, weights_initializer=cu.RandomUniformInitializer(low, high), grad_reduction='mean',
        # biases_initializer=cu.RandomUniformInitializer(-0.2, 0.2),
    )
    sequential = cm.Sequential([dense1, activation1, dense2, activation2, dense3])
    return dense1, activation1, dense2, activation2, dense3, sequential


def test_separated(func=arange_square_data, lr=0.001):
    dense1, activation1, dense2, activation2, dense3, sequential = __generate_layers()
    loss_function = cm.MSELoss(reduction='mean')
    optimizer = cm.SGD(lr=lr)

    x, y, train_dataset, accuracy_precision = generate_dataset(func)
    for epoch in range(5001):

        # Perform a forward pass of our training data through this layer
        dense1.forward(x)

        # Perform a forward pass through activation function
        # takes the output of first dense layer here
        activation1.forward(dense1.output)

        # Perform a forward pass through second Dense layer
        # takes outputs of activation function
        # of first layer as inputs
        dense2.forward(activation1.output)

        # Perform a forward pass through activation function
        # takes the output of second dense layer here
        activation2.forward(dense2.output)

        # Perform a forward pass through third Dense layer
        # takes outputs of activation function of second layer as inputs
        dense3.forward(activation2.output)

        data_loss = loss_function(dense3.output, y)

        loss = data_loss
        predictions = dense3.output
        accuracy = np.mean(np.absolute(predictions - y) <
                           accuracy_precision)

        if not epoch % 10:
            print(f'epoch: {epoch}, ' +
                  f'acc: {accuracy:.4f}, ' +
                  f'loss: {loss.item():.4f} (' +
                  f'data_loss: {data_loss.item():.4f}, ' +
                  # f'reg_loss: {regularization_loss:.8f}), ' +
                  f'lr: {optimizer.current_lr}')

        # Backward pass
        dvals = loss_function.backward(dense3.output, y)
        dvals = dense3.backward(dvals)
        dvals = activation2.backward(dvals)
        dvals = dense2.backward(dvals)
        dvals = activation1.backward(dvals)
        dense1.backward(dvals)

        # Update weights and biases
        optimizer.update([dense1, dense2, dense3])


def test_sequential(func=arange_square_data, lr=0.001):
    dense1, activation1, dense2, activation2, dense3, sequential = __generate_layers()
    loss_function = cm.MSELoss(reduction='mean')
    optimizer = cm.SGD(lr=lr)

    x, y, train_dataset, accuracy_precision = generate_dataset(func)
    for epoch in range(5001):

        # Forward pass on the sequential layer
        y_hat = sequential.forward(x)
        data_loss = loss_function(y_hat, y)

        loss = data_loss
        predictions = sequential.output
        accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)

        if not epoch % 10:
            print(f'epoch: {epoch}, ' +
                  f'acc: {accuracy:.4f}, ' +
                  f'loss: {loss.item():.4f} (' +
                  f'data_loss: {data_loss.item():.4f}, ' +
                  # f'reg_loss: {regularization_loss:.8f}), ' +
                  f'lr: {optimizer.current_lr}')

        # Backward pass
        dvals = loss_function.backward(predictions, y)
        sequential.backward(dvals)

        # Update weights and biases
        optimizer.update(sequential)


def test_sequential_minibatch(n_epochs=5000, mb_size=1, func=arange_square_data, lr=0.001):
    dense1, activation1, dense2, activation2, dense3, sequential = __generate_layers()
    loss_function = cm.MSELoss(reduction='mean')
    optimizer = cm.SGD(lr=lr)

    x, y, train_dataset, accuracy_precision = generate_dataset(func)
    mb_num = math.ceil(N_SAMPLES/mb_size)
    for epoch in range(n_epochs):
        epoch_loss = np.zeros(mb_num)
        for i in range(mb_num):
            start, end = i * mb_num, min((i+1) * mb_num, N_SAMPLES)
            x_mb, y_mb = x[start:end], y[start:end]
            # Forward pass on the sequential layer
            sequential.forward(x_mb)
            data_loss = loss_function(sequential.output, y_mb)

            loss = data_loss
            predictions = sequential.output
            epoch_loss[i] = loss
            # accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)

            # Backward pass
            dvals = loss_function.backward(predictions, y_mb)
            sequential.backward(dvals)

            # Update weights and biases
            optimizer.update(sequential)

        epoch_mean_loss = np.mean(epoch_loss)
        if not epoch % 1:
            print(f'epoch: {epoch}, ' +
                  # f'minibatch losses: {epoch_loss}',
                  # f'acc: {accuracy:.4f}, ' +
                  f'loss: {epoch_mean_loss.item():.8f} (' +
                  f'data_loss: {epoch_mean_loss.item():.8f}, ' +
                  # f'reg_loss: {regularization_loss:.8f}), ' +
                  f'lr: {optimizer.current_lr}')


def test_sequential_minibatch_dataset(n_epochs=5000, mb_size=1, epoch_shuffle=True, func=arange_square_data, lr=0.001):
    dense1, activation1, dense2, activation2, dense3, sequential = __generate_layers()
    loss_function = cm.MSELoss(reduction='mean')
    optimizer = cm.SGD(lr=lr)
    x, y, train_dataset, accuracy_precision = generate_dataset(func)
    mb_num = math.ceil(N_SAMPLES/mb_size)
    train_dataloader = DataLoader(train_dataset, batch_size=mb_size, shuffle=epoch_shuffle)
    train_dataloader.before_cycle()
    for epoch in range(n_epochs):
        epoch_loss = np.zeros(mb_num)
        train_dataloader.before_epoch()
        for i in range(mb_num):
            x_mb, y_mb = next(train_dataloader)
            # Forward pass on the sequential layer
            sequential.forward(x_mb)
            data_loss = loss_function(sequential.output, y_mb)

            loss = data_loss
            predictions = sequential.output
            epoch_loss[i] = loss
            # accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)

            # Backward pass
            dvals = loss_function.backward(predictions, y_mb)
            sequential.backward(dvals)

            # Update weights and biases
            optimizer.update(sequential)

        epoch_mean_loss = np.mean(epoch_loss)
        if not epoch % 1:
            print(f'epoch: {epoch}, ' +
                  f'loss: {epoch_mean_loss.item():.8f}' +
                  f'lr: {optimizer.current_lr}')
        train_dataloader.after_epoch()
    train_dataloader.after_cycle()


def test_fully_connected_minibatch_model(
        n_epochs=5000, mb_size=1, epoch_shuffle=True, func=arange_square_data, lr=0.001, *args, **kwargs,
):
    loss_function = cm.MSELoss(reduction='mean')
    optimizer = cm.SGD(lr=lr)
    # Generate train dataset
    x, y, train_dataset, accuracy_precision = generate_dataset(func)

    # Generate validation dataset
    args = () if args is None else args
    kwargs = {} if args is None else kwargs
    x_eval, y_eval = func(samples=N_SAMPLES//5, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, *args, **kwargs)
    eval_dataset = ArrayDataset(x_eval, y_eval)

    # Generate dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=mb_size, shuffle=epoch_shuffle)
    eval_dataloader = DataLoader(eval_dataset, batch_size=N_SAMPLES//5)
    model = cm.Model([
        cm.Dense(
            INPUT_DIM, 64, cm.ReLU(), weights_initializer=cu.RandomUniformInitializer(-1.0, 1.0),
            grad_reduction='mean'),
        cm.Dense(
            64, 64, cm.ReLU(), weights_initializer=cu.RandomUniformInitializer(-1.0, 1.0),
            grad_reduction='mean'),
        cm.Linear(64, OUTPUT_DIM, weights_initializer=cu.RandomUniformInitializer(-1.0, 1.0),
                  grad_reduction='mean')
    ])
    # Use Model class for training and epoch losses recording
    model.compile(optimizer=optimizer, loss=loss_function)
    train_epoch_losses, eval_epoch_losses, optimizer_state = model.train(train_dataloader, eval_dataloader,
                                                                         n_epochs=n_epochs)
    for epoch, (epoch_tr_loss, epoch_ev_loss, optim_state) in \
            enumerate(zip(train_epoch_losses, eval_epoch_losses, optimizer_state)):
        print(f'epoch: {epoch} ' +
              f'(tr_loss: {epoch_tr_loss.item():.8f}, ' +
              f'ev_loss: {epoch_ev_loss.item():.8f}), ' +
              f'optim_state: {optim_state}')

    plot_losses(0, train_epoch_losses, eval_epoch_losses)
