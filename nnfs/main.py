import numpy as np

from ActivationFunctions import ReLU, TanH
from Layers import Dense
from LossFunctions import MeanSquaredError
from LossRegularization import RegL1
from Optimizers import SGD

import timeit

np.random.seed(10)
X = np.random.rand(100, 2)
Y = [(x[0] * x[0] + 2 * x[1] + 0) for x in X]
Y = np.reshape(Y, (100, 1))

# Create Dense layer with 2 input features and 64 output values
# dense1 = Dense(X.shape[1], 32, "tanh", regularization=RegL1(kernel_regularizer=1e-4, bias_regularizer=1e-3))
dense1 = Dense(32,activationFunName= "tanh")

# Create ReLU activation (to be used with Dense layer):
# activation1 = TanH()

# Create second Dense layer with 64 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Dense(1)

optimizer = SGD(learning_rate=0.03, decay=0.00, momentum=0.8)

lossFun = MeanSquaredError()

start = timeit.default_timer()

for epoch in range(1501):
    dense1.forward(X,True)

    dense2.forward(dense1.output,True)
    data_loss = lossFun.forward(dense2.output, Y)

    a = dense1.regularization_loss()
    b = dense2.regularization_loss()
    regularization_loss = a + b
    # regularization_loss = dense1.regularization_loss() + dense2.regularization_loss()

    loss = data_loss + regularization_loss

    predictions = dense2.output

    if epoch % 50 == 0:
        print(f"epoch:{epoch}, loss:{loss}")

    # backward
    lossFun.backward(predictions, Y)
    dense2.backward(lossFun.dinputs)
    dense1.backward(dense2.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

stop = timeit.default_timer()

print('Time: ', stop - start)
