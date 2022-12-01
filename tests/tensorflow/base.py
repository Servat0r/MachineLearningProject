import tensorflow as tf
from tests.utils import *


layers = tf.keras.layers
tf.keras.utils.set_random_seed(SEED)


def keras_test_fully_connected_minibatch_model(
    n_epochs=5000, mb_size=1, epoch_shuffle=True, func=arange_square_data, lr=0.001, *args, **kwargs,
):
    loss_function = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    # Generate train dataset
    x, y, train_dataset, = generate_dataset(func)

    # Generate validation dataset
    args = () if args is None else args
    kwargs = {} if args is None else kwargs
    x_eval, y_eval = func(samples=N_SAMPLES//5, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, *args, **kwargs)

    # Generate dataloaders
    initializer = tf.keras.initializers.RandomUniform(-0.7, 0.7)
    model = tf.keras.Sequential()
    model.add(layers.Input((1, INPUT_DIM), batch_size=mb_size))
    model.add(layers.Dense(HIDDEN_SIZE, activation='tanh', kernel_initializer=initializer))
    model.add(layers.Dense(HIDDEN_SIZE, activation='tanh', kernel_initializer=initializer))
    model.add(layers.Dense(OUTPUT_DIM, activation='linear', kernel_initializer=initializer))
    # Use Model class for training and epoch losses recording
    model.compile(optimizer=optimizer, loss=loss_function)
    history = model.fit(x, y, validation_data=(x_eval, y_eval), batch_size=mb_size,
                        epochs=n_epochs, verbose=1, shuffle=epoch_shuffle,
                        callbacks=tf.keras.callbacks.History())
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()
    # plot_losses(0, train_epoch_losses, eval_epoch_losses)


if __name__ == '__main__':
    keras_test_fully_connected_minibatch_model(n_epochs=250, mb_size=10, epoch_shuffle=True, lr=1e-3)
    keras_test_fully_connected_minibatch_model(n_epochs=250, mb_size=10, epoch_shuffle=False, lr=1e-3)
