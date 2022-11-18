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

    # todo maybe we can handle automatic creation of dataloder based on n_epochs
    def compile(self, optimizer: Optimizer, loss: Loss, metrics=None):
        """
        Configures the model for training.
        """
        # todo create metrics (accuracy, mse/abs/mee, timing, ram usage)!
        self.optimizer = optimizer
        # auto-wrap loss into a regularization one
        self.loss = RegularizedLoss(loss, layers=self.layers)
        # todo add metrics handling
        self.layers.set_to_train()

    def train(
            self, train_dataloader: DataLoader, eval_dataloader: DataLoader = None, n_epochs: int = 1,
            train_epoch_losses: np.ndarray = None, eval_epoch_losses: np.ndarray = None,
            optimizer_state: list = None,
    ):
        eval_exists = eval_dataloader is not None
        train_epoch_losses = train_epoch_losses if train_epoch_losses is not None else np.zeros(n_epochs)
        eval_epoch_losses = eval_epoch_losses if eval_epoch_losses is not None else np.zeros(n_epochs)
        optimizer_state = optimizer_state if optimizer_state is not None else []
        mb_num = train_dataloader.get_batch_num()
        train_mb_losses = np.zeros(mb_num)

        # Callbacks before training cycle
        train_dataloader.before_cycle()
        if eval_exists:
            eval_dataloader.before_cycle()

        for epoch in range(n_epochs):
            # Callbacks before training epoch
            self.layers.set_to_train()
            train_dataloader.before_epoch()
            train_mb_losses.fill(0.)
            for mb in range(mb_num):

                mb_data = next(train_dataloader)
                if mb_data is None:
                    break
                input_mb, target_mb = mb_data[0], mb_data[1]
                y_hat = self.layers(input_mb)
                data_loss_val, reg_loss_val = self.loss(y_hat, target_mb)
                optim_log_dict = self.optimizer.to_log_dict()
                print(
                    f'[Epoch {epoch}, Minibatch {mb}]{{',
                    f'\tloss values: (data = {data_loss_val.item():.8f}, regularization = {reg_loss_val.item():.8f})',
                    f'\toptimizer state: {optim_log_dict}',
                    f'}}',
                    sep='\n',
                )
                train_mb_losses[mb] = np.mean(data_loss_val + reg_loss_val, axis=0).item()
                # Backward of loss and hidden layers
                dvals = self.loss.backward(y_hat, target_mb)
                self.layers.backward(dvals)
                self.optimizer.update(self.layers)
            train_dataloader.after_epoch()
            train_epoch_losses[epoch] = np.mean(train_mb_losses).item()

            if eval_exists:
                self.layers.set_to_eval()
                eval_dataloader.before_epoch()
                input_eval, target_eval = next(eval_dataloader)
                y_hat = self.layers(input_eval)
                data_loss_val, reg_loss_val = self.loss(y_hat, target_eval)
                optim_log_dict = self.optimizer.to_log_dict()
                print(
                    f'[Epoch {epoch}]{{',
                    f'\tloss values: (data = {data_loss_val.item():.8f}, regularization = {reg_loss_val.item():.8f})',
                    f'\toptimizer state: {optim_log_dict}',
                    f'}}',
                    sep='\n',
                )
                eval_dataloader.after_epoch()
                eval_epoch_losses[epoch] = np.mean(data_loss_val + reg_loss_val, axis=0)
                optimizer_state.append(optim_log_dict)

        train_dataloader.after_cycle()
        if eval_exists:
            eval_dataloader.after_cycle()
        return train_epoch_losses, eval_epoch_losses, optimizer_state

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
