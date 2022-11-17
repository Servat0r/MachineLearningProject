# Comments below are copied from 'two_layers_nn_sin_sgd.py'. In this testcase, the same data
# are used but layers, optimizers etc. are encapsulated in a 'Model' object for testing it,
# and Datasets and DataLoaders are also used.

# An example of test employing a two fully-connected layers Neural Network for regression
# Objective function is sin(x1+...+xn), regression loss is Mean Square Error (no regularization)
# Here we use multiple epochs with a randomly generated set of data and we "simulate" minibatches
# (there is no actual shuffle of training data and they are taken ALWAYS in the same order)
import unittest
import math

import core.utils as cu
from core.utils import np
import core.functions as cf
import core.data as cd
import core.modules as cm

INPUT_DIM = 1
HIDDEN_LAYER_SIZE = 2
TRAIN_BATCH_SIZE = 20000
EVAL_BATCH_SIZE = 2000
TRAIN_MB_SIZE = 1000
N_EPOCHS = 100
SEED = 10
FACTOR = 100.
NOISE_FACTOR = 1.5
L1_LAMBDA = 0.001


class MyTestCase(unittest.TestCase):

    def __init__(
            self, methodName: str = 'runTest', input_dim: int = INPUT_DIM, train_batch_size: int = TRAIN_BATCH_SIZE,
            eval_batch_size: int = EVAL_BATCH_SIZE, train_mb_size: int = TRAIN_MB_SIZE, n_epochs: int = N_EPOCHS,
            seed: int = SEED, hidden_layer_size: int = 10,
    ):
        super(MyTestCase, self).__init__(methodName)
        self.input_dim = input_dim
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.train_mb_size = train_mb_size
        self.n_epochs = n_epochs
        self.seed = seed
        self.hidden_layer_size = hidden_layer_size
        self.model = None
        self.mb_num = math.ceil(self.train_batch_size / self.train_mb_size)
        if self.seed is not None:
            np.random.seed(self.seed)

    def setUp(self) -> None:
        self.model = cm.Model(
            [
                cm.FullyConnectedLayer(self.input_dim, self.hidden_layer_size, func=cf.tanh,
                                       initializer=cu.RandomNormalDefaultInitializer(zero_bias=True)),
                                       # initializer=cu.RandomUniformInitializer(-1.0, 1.0, zero_bias=True)),
                cm.FullyConnectedLayer(self.hidden_layer_size, 1, func=cf.tanh,
                                       initializer=cu.RandomNormalDefaultInitializer(zero_bias=True)),
                                       # initializer=cu.RandomUniformInitializer(-1.0, 1.0, zero_bias=True))
            ],
            # regularizers={cm.L1Regularizer(l1_lambda=L1_LAMBDA)},
        )

        x_train = FACTOR * np.random.randn(self.train_batch_size, 1, self.input_dim)  # 2000 inputs of dimension 100
        # y_train = np.square(x_train)
        y_train = np.sin(np.sum(x_train, axis=2)) + NOISE_FACTOR / FACTOR * np.random.randn(self.train_batch_size, 1)
        x_train += NOISE_FACTOR * np.random.randn(self.train_batch_size, 1, self.input_dim)
        y_train = np.reshape(y_train, (self.train_batch_size, 1, 1))
        # y = sin(x1+...+xn) + random (gaussian) noise

        x_eval = FACTOR * np.random.randn(self.eval_batch_size, 1, self.input_dim)
        # y_eval = np.square(x_eval)
        y_eval = np.sin(np.sum(x_eval, axis=2)) + NOISE_FACTOR / FACTOR * np.random.randn(self.eval_batch_size, 1)
        x_eval += NOISE_FACTOR * np.random.randn(self.eval_batch_size, 1, self.input_dim)
        y_eval = np.reshape(y_eval, (self.eval_batch_size, 1, 1))

        # Create and configure datasets and dataloaders
        self.train_dataset = cd.ArrayDataset(x_train, y_train)
        self.eval_dataset = cd.ArrayDataset(x_eval, y_eval)
        self.train_dataloader = cd.DataLoader(
            self.train_dataset, batch_size=self.train_mb_size, shuffle=True, log_to='train_dataloder_log.json',
        )
        self.eval_dataloader = cd.DataLoader(
            self.eval_dataset, batch_size=self.eval_batch_size, shuffle=False, log_to='eval_dataloder_log.json',
        )

    def tearDown(self) -> None:
        self.model = None
        self.train_dataset, self.eval_dataset, self.train_dataloader, self.eval_dataloader = None, None, None, None

    def test_main(self):
        self.model.compile(
            optimizer=cm.SGD(momentum=0.9),
            loss=cm.MSELoss(),
        )
        train_epoch_losses, eval_epoch_losses = self.model.train(
            self.train_dataloader, self.eval_dataloader, n_epochs=self.n_epochs,
        )
        print(
            f"[After Training]: ", f"Average Training and Evaluation MSE Losses per epoch: ",
            str([(trloss, evloss) for trloss, evloss in zip(train_epoch_losses, eval_epoch_losses)])
        )


if __name__ == '__main__':
    test = MyTestCase(seed=SEED, n_epochs=2, train_batch_size=20000, train_mb_size=200)
    test.run()
