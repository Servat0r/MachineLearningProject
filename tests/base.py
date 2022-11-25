from __future__ import annotations
from .utils import *
import core.utils as cu
from core.data import DataLoader
import core.modules as cm


def __generate_layers(low=-1.0, high=1.0):
    dense1 = cm.Linear(
        INPUT_DIM, HIDDEN_SIZE,
        weights_initializer=cu.RandomUniformInitializer(low, high), grad_reduction='mean',
        # biases_initializer=cu.RandomUniformInitializer(-0.2, 0.2),
    )
    activation1 = cm.Tanh()
    dense2 = cm.Linear(
        HIDDEN_SIZE, HIDDEN_SIZE, weights_initializer=cu.RandomUniformInitializer(low, high), grad_reduction='mean',
        # biases_initializer=cu.RandomUniformInitializer(-0.2, 0.2),
    )
    activation2 = cm.Tanh()
    dense3 = cm.Linear(
        HIDDEN_SIZE, OUTPUT_DIM, weights_initializer=cu.RandomUniformInitializer(low, high), grad_reduction='mean',
        # biases_initializer=cu.RandomUniformInitializer(-0.2, 0.2),
    )
    return dense1, activation1, dense2, activation2, dense3


def test_separated(func=arange_square_data, lr=0.001):
    dense1, activation1, dense2, activation2, dense3 = __generate_layers()
    loss_function = cm.MSELoss(reduction='mean')
    optimizer = cm.SGD(lr=lr)

    x, y, train_dataset, accuracy_precision = generate_dataset(func)
    for epoch in range(5001):
        optimizer.before_epoch()

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
        optimizer.after_epoch()


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
            INPUT_DIM, HIDDEN_SIZE, cm.Tanh(), weights_initializer=cu.RandomUniformInitializer(-1.0, 1.0),
            grad_reduction='mean'),
        cm.Dense(
            HIDDEN_SIZE, HIDDEN_SIZE, cm.Tanh(), weights_initializer=cu.RandomUniformInitializer(-1.0, 1.0),
            grad_reduction='mean'),
        cm.Linear(HIDDEN_SIZE, OUTPUT_DIM, weights_initializer=cu.RandomUniformInitializer(-1.0, 1.0),
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
