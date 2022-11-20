import numpy as np
from Layers import Dense
from LossFunctions import MeanSquaredError
from LossRegularization import RegL1
from model import Model
from Optimizers import SGD

import timeit

np.random.seed(10)
X = np.random.rand(100, 2)
y = [(x[0] * x[0] + 2 * x[1] + 0) for x in X]
y = np.reshape(y, (100, 1))

model = Model()
model.add(Dense(32, activationFunName="tanh",regularization=RegL1(kernel_regularizer=1e-4, bias_regularizer=1e-4)))
model.add(Dense(1))
model.compile(lossname="mse", optimizer=SGD(learning_rate=0.03, decay=0.00, momentum=0.8))

start = timeit.default_timer()

model.train(X, y, epochs=1500, batch_size=10, log_every=50)

stop = timeit.default_timer()
print('Time: ', stop - start)

