# A test with CUP data for checking correctness
from __future__ import annotations
import math
import tensorflow as tf
from tests.utils import *
from core.utils.types import *
from core.data import *
from core.model_selection import Holdout
# from core.transforms import StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import keras.backend as K


layers = tf.keras.layers


def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def keras_test_cup_once(
        use_internal_test_set=True, directory_path: str = '../datasets/cup',
        internal_test_set_size=0.2, shuffle_once=True, scaler=StandardScaler(), dtype=np.float32,
):
    # Read cup dataset
    train_data, train_targets, int_test_set_data, int_test_set_targets, cup_test_set_data = read_cup(
        use_internal_test_set, directory_path, internal_test_set_size, shuffle_once, dtype=dtype
    )

    # Use Holdout split once
    eval_data, eval_targets = None, None
    cross_validator = Holdout()
    for train_values, eval_values in cross_validator.split(train_data, train_targets, shuffle=True, random_state=0,
                                                           validation_split_percentage=0.2):
        train_data, train_targets = train_values
        eval_data, eval_targets = eval_values

    # initializer = tf.keras.initializers.RandomUniform(-0.05, 0.05)
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(9,)))
    fan_in_1, fan_in_2, fan_in_3 = 1 / math.sqrt(9), 1 / math.sqrt(16), 1 / math.sqrt(8)
    print('Fan-Ins:', fan_in_1, fan_in_2, fan_in_3)
    model.add(
        layers.Dense(
            16, activation='sigmoid',
            kernel_initializer=tf.keras.initializers.RandomUniform(-fan_in_1, fan_in_1),
            kernel_regularizer=tf.keras.regularizers.L2(1e-4), bias_regularizer=tf.keras.regularizers.L2(1e-4),
        ),
    )
    model.add(
        layers.Dense(
            8, activation='sigmoid',
            kernel_initializer=tf.keras.initializers.RandomUniform(-fan_in_2, fan_in_2),
            kernel_regularizer=tf.keras.regularizers.L2(1e-4), bias_regularizer=tf.keras.regularizers.L2(1e-4),
        ),
    )
    model.add(
        layers.Dense(
            2, activation='linear',
            kernel_initializer=tf.keras.initializers.RandomUniform(-fan_in_3, fan_in_3),
            kernel_regularizer=tf.keras.regularizers.L2(1e-4), bias_regularizer=tf.keras.regularizers.L2(1e-4),
        ),
    )

    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.4)
    loss = tf.keras.losses.MeanSquaredError()

    model.compile(
        optimizer, loss,
        metrics=[
            tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError(), euclidean_distance_loss
        ]
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(min_delta=1e-3, patience=100, restore_best_weights=True)
    history = model.fit(
        train_data, train_targets, batch_size=16, epochs=2000,
        validation_data=(eval_data, eval_targets), shuffle=True,
        callbacks=[early_stopping]
    )
    if early_stopping.stopped_epoch is not None and early_stopping.stopped_epoch > 0:
        stop_epoch = early_stopping.stopped_epoch
    else:
        stop_epoch = 2000
    keras_plot_history(0, history, stop_epoch)
    int_ts_predicted = model.predict(int_test_set_data)
    means = np.mean(np.abs(int_ts_predicted - int_test_set_targets), axis=0)
    print(f'Mean values predicted: {means}')


if __name__ == '__main__':
    keras_test_cup_once()
