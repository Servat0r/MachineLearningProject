import core.modules as cm
import core.utils as cu
from core.utils import np
from core.data import ArrayDataset, DataLoader
import math


INPUT_DIM = 10
OUTPUT_DIM = 2
N_SAMPLES = 1000
SEED = 10

np.random.seed(SEED)


# Create dataset
# ---------------------- ARANGE-BASED GENERATORS ----------------------------------
def arange_sine_data(samples=1000, input_dim=1, output_dim=1, start=0):
    stop = samples * input_dim + start
    x = np.arange(start=start, stop=stop).reshape((samples, 1, input_dim)) / stop
    y = np.sin(np.sum(x, axis=-1))
    z = np.zeros((samples, output_dim))
    for i in range(samples):
        for j in range(output_dim):
            z[i, j] = y[i]
    z = z.reshape((samples, 1, output_dim))
    return x, z


def arange_square_data(samples=1000, input_dim=1, output_dim=1, start=0):
    stop = samples * input_dim + start
    x = np.arange(start=start, stop=stop).reshape((samples, 1, input_dim)) / stop
    y = np.square(np.sum(x, axis=-1))
    z = np.zeros((samples, output_dim))
    for i in range(samples):
        for j in range(output_dim):
            z[i, j] = y[i]
    z = z.reshape((samples, 1, output_dim))
    return x, z


def arange_sqrt_data(samples=1000, input_dim=1, output_dim=1, start=0):
    stop = samples * input_dim + start
    x = np.arange(start=start, stop=stop).reshape((samples, 1, input_dim)) / stop
    y = np.sqrt(np.abs(np.sum(x, axis=-1)))
    z = np.zeros((samples, output_dim))
    for i in range(samples):
        for j in range(output_dim):
            z[i, j] = y[i]
    z = z.reshape((samples, 1, output_dim))
    return x, z


# ----------------------- NORMAL DISTRIBUTION - BASED GENERATORS ------------------
def randn_sine_data(samples=1000, input_dim=1, output_dim=1, normalize=True):
    x = np.random.randn(samples * input_dim).reshape((samples, 1, input_dim))
    if normalize:
        x /= np.max(x)  # normalization
    y = np.sin(np.sum(x, axis=-1))
    z = np.zeros((samples, output_dim))
    for i in range(samples):
        for j in range(output_dim):
            z[i, j] = y[i]
    z = z.reshape((samples, 1, output_dim))
    return x, z


def randn_square_data(samples=1000, input_dim=1, output_dim=1, normalize=True):
    x = np.random.randn(samples * input_dim).reshape((samples, 1, input_dim))
    if normalize:
        x /= np.max(x)  # normalization
    y = np.square(np.sum(x, axis=-1))
    z = np.zeros((samples, output_dim))
    for i in range(samples):
        for j in range(output_dim):
            z[i, j] = y[i]
    z = z.reshape((samples, 1, output_dim))
    return x, z


def randn_sqrt_data(samples=1000, input_dim=1, output_dim=1, normalize=True):
    x = np.random.randn(samples * input_dim).reshape((samples, 1, input_dim))
    if normalize:
        x /= np.max(x)  # normalization
    y = np.sqrt(np.abs(np.sum(x, axis=-1)))
    z = np.zeros((samples, output_dim))
    for i in range(samples):
        for j in range(output_dim):
            z[i, j] = y[i]
    z = z.reshape((samples, 1, output_dim))
    return x, z


# ---------------------- SAMPLE NN (WITHOUT model) --------------------------------------
dense1 = cm.Linear(INPUT_DIM, 64, cu.RandomUniformInitializer(-1.0, 1.0), grad_reduction='mean')
activation1 = cm.ReLU()
dense2 = cm.Linear(64, 64, cu.RandomUniformInitializer(-1.0, 1.0), grad_reduction='mean')
activation2 = cm.ReLU()
dense3 = cm.Linear(64, OUTPUT_DIM, cu.RandomUniformInitializer(-1.0, 1.0), grad_reduction='mean')
# we should need a 'linear activation'
sequential = cm.Sequential([dense1, activation1, dense2, activation2, dense3])
# ---------------------- SAMPLE MODEL ---------------------------------------------------
model = cm.Model(sequential)

# ----------------------- LOSS AND OPTIMIZER --------------------------------------------
loss_function = cm.MSELoss(reduction='mean')
optimizer = cm.SGD(lr=0.005)


def generate_dataset(func, samples=N_SAMPLES, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, *args, **kwargs):

    args = () if args is None else args
    kwargs = {} if kwargs is None else kwargs
    x, y = func(samples=samples, input_dim=input_dim, output_dim=output_dim, *args, **kwargs)
    x = x.reshape((x.shape[0], x.shape[1], INPUT_DIM))
    y = y.reshape((y.shape[0], y.shape[1], OUTPUT_DIM))
    train_dataset = ArrayDataset(x, y)
    accuracy_precision = np.std(y) / 250

    return x, y, train_dataset, accuracy_precision


def test_separated(func=arange_square_data):
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


def test_sequential(func=arange_square_data):
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


def test_sequential_minibatch(n_epochs=5000, mb_size=1, func=arange_square_data):
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


def test_sequential_minibatch_dataset(n_epochs=5000, mb_size=1, epoch_shuffle=True,
                                      func=arange_square_data, use_model=False):
    x, y, train_dataset, accuracy_precision = generate_dataset(func)
    mb_num = math.ceil(N_SAMPLES/mb_size)
    train_dataloader = DataLoader(train_dataset, batch_size=mb_size, shuffle=epoch_shuffle)

    if use_model:
        # Use Model class for training and epoch losses recording
        model.compile(optimizer=optimizer, loss=loss_function)
        train_epoch_losses, _ = model.train(train_dataloader, n_epochs=n_epochs)
        for epoch, epoch_loss in enumerate(train_epoch_losses):
            print(f'epoch: {epoch}, ' +
                  f'loss: {epoch_loss.item():.8f}')
    else:
        # Manual training and loss recording
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
        n_epochs=5000, mb_size=1, epoch_shuffle=True, func=arange_square_data, *args, **kwargs,
):
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


def test_fully_connected_minibatch_model_with_regularizations(
        n_epochs=5000, mb_size=1, epoch_shuffle=True, func=arange_square_data,
        l1_regularizer=0., l2_regularizer=0., *args, **kwargs,
):
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
            INPUT_DIM, 64, cm.ReLU(),
            weights_initializer=cu.RandomUniformInitializer(-1.0, 1.0), grad_reduction='mean',
            biases_initializer=cu.RandomUniformInitializer(-1.0, 1.0),
            weights_regularizer=cm.L1L2Regularizer(l1_lambda=l1_regularizer, l2_lambda=l2_regularizer),
            biases_regularizer=cm.L1L2Regularizer(l1_lambda=l1_regularizer, l2_lambda=l2_regularizer),
        ),
        cm.Dense(
            64, 64, cm.ReLU(),
            weights_initializer=cu.RandomUniformInitializer(-1.0, 1.0), grad_reduction='mean',
            biases_initializer=cu.RandomUniformInitializer(-1.0, 1.0),
            weights_regularizer=cm.L1L2Regularizer(l1_lambda=l1_regularizer, l2_lambda=l2_regularizer),
            biases_regularizer=cm.L1L2Regularizer(l1_lambda=l1_regularizer, l2_lambda=l2_regularizer),
        ),
        cm.Linear(
            64, OUTPUT_DIM,
            weights_initializer=cu.RandomUniformInitializer(-1.0, 1.0), grad_reduction='mean',
            biases_initializer=cu.RandomUniformInitializer(-1.0, 1.0),
            weights_regularizer=cm.L1L2Regularizer(l1_lambda=l1_regularizer, l2_lambda=l2_regularizer),
            biases_regularizer=cm.L1L2Regularizer(l1_lambda=l1_regularizer, l2_lambda=l2_regularizer),
        )
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


def test_fc_minibatch_model_with_regularizations_lrscheduler(
        n_epochs=5000, mb_size=1, epoch_shuffle=True, func=arange_square_data, lr=0.005, momentum=0.,
        lr_scheduler: cm.Scheduler = None, l1_regularizer=0., l2_regularizer=0., *args, **kwargs,
):
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
            INPUT_DIM, 64, cm.ReLU(),
            weights_initializer=cu.RandomUniformInitializer(-1.0, 1.0), grad_reduction='mean',
            # biases_initializer=cu.RandomUniformInitializer(-1.0, 1.0),
            weights_regularizer=cm.L1L2Regularizer(l1_lambda=l1_regularizer, l2_lambda=l2_regularizer),
            biases_regularizer=cm.L1L2Regularizer(l1_lambda=l1_regularizer, l2_lambda=l2_regularizer),
        ),
        cm.Dense(
            64, 64, cm.ReLU(),
            weights_initializer=cu.RandomUniformInitializer(-1.0, 1.0), grad_reduction='mean',
            # biases_initializer=cu.RandomUniformInitializer(-1.0, 1.0),
            weights_regularizer=cm.L1L2Regularizer(l1_lambda=l1_regularizer, l2_lambda=l2_regularizer),
            biases_regularizer=cm.L1L2Regularizer(l1_lambda=l1_regularizer, l2_lambda=l2_regularizer),
        ),
        cm.Linear(
            64, OUTPUT_DIM,
            weights_initializer=cu.RandomUniformInitializer(-1.0, 1.0), grad_reduction='mean',
            # biases_initializer=cu.RandomUniformInitializer(-1.0, 1.0),
            weights_regularizer=cm.L1L2Regularizer(l1_lambda=l1_regularizer, l2_lambda=l2_regularizer),
            biases_regularizer=cm.L1L2Regularizer(l1_lambda=l1_regularizer, l2_lambda=l2_regularizer),
        )
    ])
    # Shadow optimizer in order to use custom lr and scheduler
    optimizer = cm.SGD(lr=lr, lr_decay_scheduler=lr_scheduler, momentum=momentum)

    # Use Model class for training and epoch losses recording
    model.compile(optimizer=optimizer, loss=loss_function)
    train_epoch_losses, eval_epoch_losses, optimizer_state = \
        model.train(train_dataloader, eval_dataloader, n_epochs=n_epochs)

    # Print results
    for epoch, (epoch_tr_loss, epoch_ev_loss, optim_state) in \
            enumerate(zip(train_epoch_losses, eval_epoch_losses, optimizer_state)):
        print(f'epoch: {epoch} ' +
              f'(tr_loss: {epoch_tr_loss.item():.8f}, ' +
              f'ev_loss: {epoch_ev_loss.item():.8f}), ' +
              f'optim_state: {optim_state}')


if __name__ == '__main__':
    # Uncomment the tests you want to execute
    # todo: test_separated, test_sequential, test_sequential_minibatch,
    # test_sequential_minibatch_dataset(..., use_model=False) are ALL based
    # on the SAME layers, so you shall run at most ONE of them (for now)

    """
    test_separated()
    test_sequential()
    test_sequential_minibatch(n_epochs=100, mb_size=100)
    test_sequential_minibatch_dataset(n_epochs=100, mb_size=100, func=randn_sqrt_data)
    test_sequential_minibatch_dataset(n_epochs=100, mb_size=100, func=randn_sqrt_data, epoch_shuffle=False)
    test_sequential_minibatch_dataset(n_epochs=100, mb_size=100, func=randn_sqrt_data, use_model=True)
    test_sequential_minibatch_dataset(
        n_epochs=100, mb_size=100, func=randn_sqrt_data, epoch_shuffle=False, use_model=True
    )
    test_fully_connected_minibatch_model(n_epochs=100, mb_size=100, func=randn_sqrt_data)
    test_fully_connected_minibatch_model(n_epochs=100, mb_size=100, func=randn_sqrt_data, epoch_shuffle=False)
    """
    test_fully_connected_minibatch_model_with_regularizations(
        n_epochs=100, mb_size=100, func=arange_sine_data, l1_regularizer=0.001, l2_regularizer=0.001,
    )
    """
    test_fully_connected_minibatch_model_with_regularizations(
        n_epochs=100, mb_size=100, func=randn_sqrt_data, epoch_shuffle=False,
        l1_regularizer=0., l2_regularizer=0.01,
    )
    test_fc_minibatch_model_with_regularizations_lrscheduler(
        n_epochs=100, mb_size=50, func=randn_sqrt_data, lr=0.01, momentum=0.,
        # max_iter = n_epochs * mb_size
        # lr_scheduler=cm.LinearDecayScheduler(start_value=0.01, end_value=0.005, max_iter=100*(N_SAMPLES//50)),
        # lr_scheduler=cm.IterBasedDecayScheduler(start_value=0.01, decay=0.001),
        lr_scheduler=cm.ExponentialDecayScheduler(start_value=0.01, alpha=0.001),
        l1_regularizer=0.001, l2_regularizer=0.001,
    )
    test_fc_minibatch_model_with_regularizations_lrscheduler(
        n_epochs=100, mb_size=50, func=randn_sqrt_data, lr=0.01, momentum=0., epoch_shuffle=False,
        # max_iter = n_epochs * mb_size
        # lr_scheduler=cm.LinearDecayScheduler(start_value=0.01, end_value=0.005, max_iter=100*(N_SAMPLES//50)),
        # lr_scheduler=cm.IterBasedDecayScheduler(start_value=0.01, decay=0.001),
        lr_scheduler=cm.ExponentialDecayScheduler(start_value=0.01, alpha=0.001),
        l1_regularizer=0.001, l2_regularizer=0.001,
    )
    test_fc_minibatch_model_with_regularizations_lrscheduler(
        n_epochs=100, mb_size=50, func=arange_sine_data, lr=0.01, momentum=0.9,
        # max_iter = n_epochs * mb_size
        # lr_scheduler=cm.LinearDecayScheduler(start_value=0.01, end_value=0.005, max_iter=100*(N_SAMPLES//50)),
        # lr_scheduler=cm.IterBasedDecayScheduler(start_value=0.01, decay=0.001),
        lr_scheduler=cm.ExponentialDecayScheduler(start_value=0.01, alpha=0.001),
        l1_regularizer=0.00001, l2_regularizer=0.00001,
        # arange_sine_data extra args for validation set
        start=N_SAMPLES,
    )
    test_fc_minibatch_model_with_regularizations_lrscheduler(
        n_epochs=100, mb_size=50, func=arange_sine_data, lr=0.01, momentum=0., epoch_shuffle=False,
        # max_iter = n_epochs * mb_size
        # lr_scheduler=cm.LinearDecayScheduler(start_value=0.01, end_value=0.005, max_iter=100*(N_SAMPLES//50)),
        # lr_scheduler=cm.IterBasedDecayScheduler(start_value=0.01, decay=0.001),
        lr_scheduler=cm.ExponentialDecayScheduler(start_value=0.01, alpha=0.001),
        l1_regularizer=0.001, l2_regularizer=0.001,
        # arange_sine_data extra args for validation set
        start=N_SAMPLES,
    )
    """
    exit(0)
