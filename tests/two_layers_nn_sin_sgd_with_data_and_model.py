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
from core.modules import WeightedLayerParameters as WLParameters

INPUT_DIM = 100
HIDDEN_LAYER_SIZE = 10
TRAIN_BATCH_SIZE = 20000
EVAL_BATCH_SIZE = 2000
TRAIN_MB_SIZE = 1000
N_EPOCHS = 4
SEED = 10


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
        self.regularizers = {cm.L1Regularizer(l1_lambda=0.01)}   # L1 regularization
        # self.mse_loss = cm.RegularizedLoss(cm.MSELoss(), regularizers=self.regularizers)
        self.mse_loss = cm.MSELoss()

    def setUp(self) -> None:
        self.model = cm.Model(
            cm.SequentialLayer([
                cm.FullyConnectedLayer(self.input_dim, self.hidden_layer_size, func=cf.tanh,
                                       initializer=cu.RandomNormalDefaultInitializer(),),
                                       # regularizers=self.regularizers),
                cm.FullyConnectedLayer(self.hidden_layer_size, 1, func=cf.tanh,
                                       initializer=cu.RandomNormalDefaultInitializer(),)
                                       # regularizers=self.regularizers),
            ])
        )
        self.sgd_optimizer = cm.SGD(
            cu.cast(set[WLParameters], self.model.get_parameters()),
            momentum=0.9,
        )

        train_base_shape = (self.train_batch_size, 1)
        eval_base_shape = (self.eval_batch_size, 1)

        x_train = 100. * np.random.randn(*train_base_shape, self.input_dim)  # 2000 inputs of dimension 100
        y_train = np.sin(np.sum(x_train, axis=2)) + np.random.randn(*train_base_shape)
        y_train = np.reshape(y_train, train_base_shape + (1,))
        # y = sin(x1+...+xn) + random (gaussian) noise

        x_eval = 100. * np.random.randn(*eval_base_shape, self.input_dim)
        y_eval = np.sin(np.sum(x_eval, axis=2))
        y_eval = np.reshape(y_eval, eval_base_shape + (1,))

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
        self.sgd_optimizer = None
        self.train_dataset, self.eval_dataset, self.train_dataloader, self.eval_dataloader = None, None, None, None

    def test_main(self):
        # total_train_mse_losses = []
        # epoch_training_mse_losses = np.zeros(self.mb_num)
        # total_eval_mse_losses = []
        self.model.compile(optimizer=self.sgd_optimizer, loss=self.mse_loss)
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
