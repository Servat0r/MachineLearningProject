import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
import timeit

from ActivationFunctions import ActivationBase

np.random.seed(10)
X_train = np.random.rand(100, 2)
y_train = [(x[0] * x[0] + 2 * x[1]+0) for x in X_train]
y_train = np.reshape(y_train, (100, 1))

initializer = tf.keras.initializers.RandomUniform(minval=-0.7, maxval=0.7)

model = tf.keras.Sequential([tf.keras.layers.Dense(32, activation="tanh", kernel_initializer=initializer,
                                                   kernel_regularizer=regularizers.L1(1e-4),
                                                   bias_regularizer=regularizers.L1(1e-4)),
                             tf.keras.layers.Dense(1)])

# model = tf.keras.Sequential([tf.keras.layers.Dense(32, activation="tanh", kernel_initializer=initializer),
#                              tf.keras.layers.Dense(1)])

model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=0.03, momentum=0.8))

start = timeit.default_timer()

history = model.fit(X_train, y_train, shuffle=False, epochs=1500, batch_size=10)

stop = timeit.default_timer()

print('Time: ', stop - start)
