# Model class: main class for building a complete Neural Network
from __future__ import annotations
from ..utils import *
from .layers import *
from .losses import *
from .optimizers import *
from ..data import *


class Model:
    """
    Base class for a Neural Network
    """
    def __init__(self, layers: Layer | Sequence[Layer]):
        self.layers = layers if isinstance(layers, SequentialLayer) else SequentialLayer(layers)
        self.regularizers = layers.get_regularizers()
        self.optimizer = None
        self.loss = None
        self.metrics = None

    @staticmethod
    @abstractmethod
    def load(fpath: str) -> Model:
        pass

    @staticmethod
    @abstractmethod
    def save(fpath: str) -> Model:
        pass

    def get_parameters(self):
        return self.layers.get_parameters()

    # todo maybe we can handle automatic creation of dataloder based on n_epochs
    def compile(self, optimizer: Optimizer, loss: Loss, metrics=None):
        """
        Configures the model for training.
        """
        # todo create metrics (accuracy, mse/abs/mee, timing, ram usage)!
        self.optimizer = optimizer
        if len(self.regularizers) > 0:
            self.loss = RegularizedLoss(loss, regularizers=self.regularizers)
        else:
            self.loss = loss
        # todo add metrics handling
        self.layers.set_to_train()

    # todo fixme Completare!
    def train(
            self, train_dataloader: DataLoader, eval_dataloader: DataLoader = None, n_epochs: int = 1,
            train_epoch_losses: np.ndarray = None, eval_epoch_losses: np.ndarray = None,
    ):
        eval_exists = eval_dataloader is not None
        train_epoch_losses = train_epoch_losses if train_epoch_losses is not None else np.zeros(n_epochs)
        eval_epoch_losses = eval_epoch_losses if eval_epoch_losses is not None else np.zeros(n_epochs)
        mb_num = train_dataloader.get_batch_num()
        train_mb_losses = np.zeros(mb_num)

        # Callbacks before training cycle
        train_dataloader.before_cycle()
        if eval_exists:
            eval_dataloader.before_cycle()

        for epoch in range(n_epochs):
            train_iter, eval_iter = iter(train_dataloader), iter(eval_dataloader)
            # Callbacks before training epoch
            self.layers.set_to_train()
            train_dataloader.before_epoch()
            train_mb_losses.fill(0.)
            for mb in range(mb_num):
                mb_data = next(train_iter)
                if mb_data is None:
                    break
                input_mb, target_mb = mb_data[0], mb_data[1]
                y_hat = self.layers(input_mb)
                loss_val = self.loss(y_hat, target_mb)
                print(f'[Epoch {epoch}, Minibatch {mb}]: loss values over {len(input_mb)} '
                      f'training examples = {loss_val}')
                train_mb_losses[mb] = loss_val
                dvals = self.loss.backward()
                self.layers.backward(dvals)
                self.optimizer.update()
            train_dataloader.after_epoch()
            train_epoch_losses[epoch] = np.mean(train_mb_losses).item()
            if eval_exists:
                self.layers.set_to_eval()
                eval_dataloader.before_epoch()
                input_eval, target_eval = next(eval_iter)
                y_hat = self.layers(input_eval)
                loss_val = self.loss(y_hat, target_eval)
                print(f'[Epoch {epoch}]: loss values over {len(input_eval)} validation examples = {loss_val}')
                eval_dataloader.after_epoch()
                eval_epoch_losses[epoch] = loss_val

        train_dataloader.after_cycle()
        if eval_exists:
            eval_dataloader.after_cycle()
        return train_epoch_losses, eval_epoch_losses

    def reset(self):
        """
        Resets the model for a fresh usage.
        """
        self.layers.set_to_eval()
        self.optimizer = None
        self.loss = None
        self.metrics = None


__all__ = [
    'Model',
]
