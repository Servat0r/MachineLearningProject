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
        self.layers = None
        self.mb_num = math.ceil(self.train_batch_size / self.train_mb_size)
        if self.seed is not None:
            np.random.seed(self.seed)
        self.regularizers = {cm.L1Regularizer(l1_lambda=0.01)}   # L1 regularization
        # self.mse_loss = cm.RegularizedLoss(cm.MSELoss(), regularizers=self.regularizers)
        self.mse_loss = cm.MSELoss()

    def setUp(self) -> None:
        self.layers = cm.SequentialLayer([
            cm.FullyConnectedLayer(self.input_dim, self.hidden_layer_size, func=cf.tanh,
                                   initializer=cu.RandomNormalDefaultInitializer(),
                                   regularizers=self.regularizers),
            cm.FullyConnectedLayer(self.hidden_layer_size, 1, func=cf.tanh,
                                   initializer=cu.RandomNormalDefaultInitializer(),
                                   regularizers=self.regularizers),
        ])
        self.sgd_optimizer = cm.SGD(
            cu.cast(set[WLParameters], self.layers.get_parameters()),
            momentum=0.9
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

        train_dataset = cd.ArrayDataset(x_train, y_train)
        eval_dataset = cd.ArrayDataset(x_eval, y_eval)
        self.train_dataloader = cd.DataLoader(train_dataset, batch_size=self.train_mb_size, shuffle=True)
        self.eval_dataloader = cd.DataLoader(eval_dataset, batch_size=self.eval_batch_size, shuffle=False)

    def tearDown(self) -> None:
        self.layers = None
        self.sgd_optimizer = None
        self.train_dataloader, self.eval_dataloader = None, None

    def test_main(self):
        total_train_mse_losses = np.zeros(self.n_epochs)
        epoch_training_mse_losses = np.zeros(self.mb_num)
        total_eval_mse_losses = np.zeros(self.n_epochs)
        print(f"[Before Training]: {self.train_batch_size} training examples of dimension {self.input_dim}")
        self.train_dataloader.before_cycle()  
        self.eval_dataloader.before_cycle()  
        for epoch in range(self.n_epochs):
            train_iter, eval_iter = iter(self.train_dataloader), iter(self.eval_dataloader)  
            self.train_dataloader.before_epoch()  
            for mb in range(self.mb_num):
                x_data, y_data = next(train_iter)
                y_hat = self.layers(x_data)  
                mse_losses = self.mse_loss(y_hat, y_data)
                acc = np.abs(y_hat - y_data)  
                print(f"[Epoch {epoch}, Minibatch {mb}]:", f"Average MSE Loss over {TRAIN_MB_SIZE} training examples =",
                      mse_losses, f"Average distance of predicted outputs from real ones =", np.mean(acc))
                epoch_training_mse_losses[mb] = mse_losses.item()
                dvals = self.mse_loss.backward()
                self.layers.backward(dvals)
                self.sgd_optimizer.update()
            total_train_mse_losses[epoch] = np.mean(epoch_training_mse_losses).item()
            self.train_dataloader.after_epoch()  
            self.eval_dataloader.before_epoch()  
            print(f"[Before Evaluating (Epoch {epoch})]: {self.eval_batch_size} test examples")
            x_eval, y_eval = next(eval_iter)
            eval_y_hat = self.layers(x_eval)  
            eval_mse_losses = self.mse_loss(eval_y_hat, y_eval)
            eval_acc = np.abs(eval_y_hat - y_eval)
            total_eval_mse_losses[epoch] = eval_mse_losses.item()
            print(f"[After Evaluating (Epoch {epoch})]:", f"Average MSE Loss over {self.eval_batch_size} examples",
                  eval_mse_losses, f"Average distance of predicted outputs from real ones =", np.mean(eval_acc))
            self.eval_dataloader.after_epoch()  
        self.train_dataloader.after_cycle()  
        self.eval_dataloader.after_cycle()  
        print(
            f"[After Training]: ", f"Average Training and Evaluation MSE Losses per epoch: ",
            str([(trloss, evloss) for trloss, evloss in zip(total_train_mse_losses, total_eval_mse_losses)])
        )


if __name__ == '__main__':
    test = MyTestCase(seed=SEED, n_epochs=2, train_batch_size=20000, train_mb_size=200)
    test.run()
