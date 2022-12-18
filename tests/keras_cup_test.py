# A test with CUP data for checking correctness
from __future__ import annotations
import tensorflow as tf
from tests.utils import *
from core.utils.types import *
from core.data import *
from core.model_selection import Holdout
# from core.transforms import StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler


layers = tf.keras.layers


def keras_test_cup_once(
        use_internal_test_set=True, directory_path: str = '../datasets/cup',
        internal_test_set_size=0.2, shuffle_once=True, scaler=StandardScaler(), dtype=np.float32,
):
    # Read cup dataset
    train_data, train_targets, int_test_set_data, int_test_set_targets, cup_test_set_data = read_cup(
        use_internal_test_set, directory_path, internal_test_set_size, shuffle_once, dtype
    )

    # Use Holdout split once
    eval_data, eval_targets = None, None
    cross_validator = Holdout()
    for train_values, eval_values in cross_validator.split(train_data, train_targets, shuffle=True, random_state=0,
                                                           validation_split_percentage=0.25):
        train_data, train_targets = train_values
        eval_data, eval_targets = eval_values

    # initializer = tf.keras.initializers.RandomUniform(-0.05, 0.05)
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(1, 9)))
    model.add(
        layers.Dense(16, activation='tanh',
                     kernel_initializer=tf.keras.initializers.RandomUniform(-0.05, 0.05))
    )
    model.add(
        layers.Dense(
            8, activation='tanh',
            kernel_initializer=tf.keras.initializers.RandomUniform(-0.05, 0.05))
    )
    model.add(
        layers.Dense(
            2, activation='linear',
            kernel_initializer=tf.keras.initializers.RandomUniform(-0.05, 0.05))
    )

    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.0)
    loss = tf.keras.losses.MeanSquaredError()

    model.compile(
        optimizer, loss, metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(min_delta=1e-4, patience=100)
    history = model.fit(
        train_data, train_targets, batch_size=8, epochs=500,
        validation_data=(eval_data, eval_targets), shuffle=True,
        callbacks=[early_stopping]
    )
    keras_plot_history(0, history, early_stopping.stopped_epoch)


if __name__ == '__main__':
    keras_test_cup_once()
