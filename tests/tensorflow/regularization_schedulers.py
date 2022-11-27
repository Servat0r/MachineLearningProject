import tensorflow as tf
from tests.tensorflow.utils import *
from tests.utils import *
import core.modules as cm


layers = tf.keras.layers
# I know that the above is awful, but for some reason PyCharm does not visualize tensorflow.keras when importing
tf.keras.utils.set_random_seed(SEED)


def __generate_model(initializer, l1_regularizer, l2_regularizer, mb_size):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(1, INPUT_DIM), batch_size=mb_size))
    model.add(layers.Dense(
        HIDDEN_SIZE, activation='tanh', kernel_initializer=initializer,
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_regularizer, l2=l2_regularizer),
        bias_regularizer=tf.keras.regularizers.L1L2(l1=l1_regularizer, l2=l2_regularizer),
    ))
    model.add(layers.Dense(
        HIDDEN_SIZE, activation='tanh', kernel_initializer=initializer,
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_regularizer, l2=l2_regularizer),
        bias_regularizer=tf.keras.regularizers.L1L2(l1=l1_regularizer, l2=l2_regularizer),
    ))
    model.add(layers.Dense(
        OUTPUT_DIM, activation='linear', kernel_initializer=initializer,
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_regularizer, l2=l2_regularizer),
        bias_regularizer=tf.keras.regularizers.L1L2(l1=l1_regularizer, l2=l2_regularizer),
    ))
    return model


def keras_test_fully_connected_minibatch_model_with_regularizations(
        n_epochs=5000, mb_size=1, epoch_shuffle=True, func=arange_square_data,
        l1_regularizer=0., l2_regularizer=0., lr=0.001, start_plot_epoch=0, *args, **kwargs,
):
    # Generate train dataset
    x, y, train_dataset, accuracy_precision = generate_dataset(func)

    # Generate validation dataset
    args = () if args is None else args
    kwargs = {} if args is None else kwargs
    x_eval, y_eval = func(samples=N_SAMPLES//5, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, *args, **kwargs)

    initializer = tf.keras.initializers.RandomUniform(-0.7, 0.7)
    model = __generate_model(initializer, l1_regularizer, l2_regularizer, mb_size)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    loss_function = tf.keras.losses.MeanSquaredError()
    # Use Model class for training and epoch losses recording
    model.compile(optimizer=optimizer, loss=loss_function)
    history = model.fit(
        x, y, batch_size=mb_size, epochs=n_epochs, verbose=1, validation_data=(x_eval, y_eval), shuffle=epoch_shuffle,
    )
    keras_plot_losses(history)


def keras_test_fully_connected_minibatch_model_with_regularizations_lrscheduler(
        n_epochs=5000, mb_size=1, epoch_shuffle=True, func=arange_square_data, lr=0.001, momentum=0.,
        lr_scheduler: cm.Scheduler = None, l1_regularizer=0., l2_regularizer=0., start_plot_epoch=0,
        *args, **kwargs,
):
    # Generate train dataset
    x, y, train_dataset, accuracy_precision = generate_dataset(func)

    # Generate validation dataset
    initializer = tf.keras.initializers.RandomUniform(-0.7, 0.7)
    args = () if args is None else args
    kwargs = {} if args is None else kwargs
    x_eval, y_eval = func(samples=N_SAMPLES//5, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, *args, **kwargs)
    model = __generate_model(initializer, l1_regularizer, l2_regularizer, mb_size)
    # Shadow optimizer in order to use custom lr and scheduler
    loss_function = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)

    # Use Model class for training and epoch losses recording
    model.compile(optimizer=optimizer, loss=loss_function)
    history = model.fit(
        x, y, batch_size=mb_size, epochs=n_epochs, verbose=1,
        callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_scheduler)],
        validation_data=(x_eval, y_eval), shuffle=epoch_shuffle,
    )
    keras_plot_losses(history)


def keras_test_fc_minibatch_model_regularization(*test_nums: int):
    if 0 in test_nums:
        keras_test_fully_connected_minibatch_model_with_regularizations(
            n_epochs=250, mb_size=10, func=arange_square_data, l1_regularizer=1e-6, l2_regularizer=1e-7,
            start_plot_epoch=0, lr=1e-4, epoch_shuffle=True,
        )
    if 1 in test_nums:
        keras_test_fully_connected_minibatch_model_with_regularizations(
            n_epochs=250, mb_size=10, func=arange_square_data, l1_regularizer=1e-6, l2_regularizer=1e-7,
            start_plot_epoch=0, lr=1e-4, epoch_shuffle=False,
        )


def keras_test_fc_minibatch_model_regularization_lrschedulers(*test_nums: int):
    if 0 in test_nums:
        keras_test_fully_connected_minibatch_model_with_regularizations_lrscheduler(
            n_epochs=100, mb_size=10, func=arange_square_data, lr=1e-4, momentum=0.9,
            # max_iter = n_epochs * mb_size
            lr_scheduler=cm.LinearDecayScheduler(
                start_value=1e-4, end_value=1e-5, max_iter=100, round_val=6,
            ),
            # lr_scheduler=cm.IterBasedDecayScheduler(start_value=0.01, decay=0.001),
            # lr_scheduler=cm.ExponentialDecayScheduler(start_value=1e-3, alpha=1e-3),
            l1_regularizer=1e-6, l2_regularizer=1e-7,
            # arange_sine_data extra args for validation set
            start=N_SAMPLES,
        )
    if 1 in test_nums:
        keras_test_fully_connected_minibatch_model_with_regularizations_lrscheduler(
            n_epochs=100, mb_size=10, func=arange_square_data, lr=1e-4, momentum=0.9, epoch_shuffle=False,
            # max_iter = n_epochs * mb_size
            lr_scheduler=cm.LinearDecayScheduler(
                start_value=1e-4, end_value=1e-5, max_iter=100, round_val=6,
            ),
            # lr_scheduler=cm.IterBasedDecayScheduler(start_value=0.01, decay=0.001),
            # lr_scheduler=cm.ExponentialDecayScheduler(start_value=1e-3, alpha=1e-3),
            l1_regularizer=1e-6, l2_regularizer=1e-7,
            # arange_sine_data extra args for validation set
            start=N_SAMPLES,
        )


if __name__ == '__main__':
    keras_test_fc_minibatch_model_regularization(0, 1)
    keras_test_fc_minibatch_model_regularization_lrschedulers(0, 1)
    exit(0)
