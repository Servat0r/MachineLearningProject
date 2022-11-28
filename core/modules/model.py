# Model class: main class for building a complete Neural Network
from __future__ import annotations
import pickle

from ..utils import *
from .layers import *
from .losses import *
from .optimizers import *
from ..metrics import *
from ..data import *


class Model:
    """
    Base class for a Neural Network
    """
    def __init__(self, layers: Layer | Sequence[Layer]):
        self.layers: Sequence[Layer] = [layers] if isinstance(layers, Layer) else list(layers)
        self.optimizer = None
        self.loss = None
        self.metrics = None
        self.length = len(self.layers)
        self.history = None  # will be filled during training
        self.__is_training = None

    def get_parameters(self) -> list[dict]:
        return [layer.get_parameters() for layer in self.layers]

    def set_parameters(self, params: list):
        for layer, param_dict in zip(self.layers, params):
            layer.set_parameters(param_dict)

    def save_parameters(self, fpath: str):
        with open(fpath, 'wb') as fp:
            pickle.dump(self.get_parameters(), fp)

    def load_parameters(self, fpath: str):
        with open(fpath, 'rb') as fp:
            self.set_parameters(pickle.load(fp))

    @staticmethod
    def load(fpath: str) -> Model:
        with open(fpath, 'rb') as fp:
            model = pickle.load(fp)
        return model

    def save(self, fpath: str, include_compile_objs=True, include_history=True):
        # Detach optimizer, loss and metrics if not requested
        loss, optim, metrics, history = None, None, None, None
        if not include_compile_objs:
            optim = self.optimizer
            loss = self.loss
            metrics = self.metrics
            self.optimizer = None
            self.loss = None
            self.metrics = None
        if not include_history:
            history = self.history
            self.history = None
        # Serialize model
        with open(fpath, 'wb') as fp:
            pickle.dump(self, fp)
        # Re-attach optimizer and loss if they were detached
        if not include_compile_objs:
            self.loss = loss
            self.optimizer = optim
            self.metrics = metrics
        if not include_history:
            self.history = history

    def set_to_train(self):
        self.__is_training = True
        for layer in self.layers:
            layer.set_to_train()

    def set_to_eval(self, detach_history=False):
        self.__is_training = False
        for layer in self.layers:
            layer.set_to_eval()
        if detach_history:
            history = self.history
            self.history = None
            return history
        return None

    def is_training(self):
        return self.__is_training

    def forward(self, x: np.ndarray):
        current_output = x
        for i in range(self.length):
            layer = self.layers[i]
            current_output = layer.forward(current_output)
        return current_output

    def backward(self, dvals: np.ndarray):
        current_dvals = dvals
        for i in range(self.length):
            layer = self.layers[self.length - 1 - i]
            current_dvals = layer.backward(current_dvals)
        return current_dvals

    def compile(self, optimizer: Optimizer, loss: Loss, metrics: Metric | Sequence[Metric] = None):
        """
        Configures the model for training.
        """
        # todo create metrics (accuracy, mse/abs/mee, timing, ram usage)!
        self.optimizer = optimizer
        # auto-wrap loss into a regularization one
        self.loss = RegularizedLoss(loss)
        metrics = [metrics] if isinstance(metrics, Metric) else metrics
        self.metrics = metrics if metrics is not None else []  # this ensures "None-safety" for the training loop
        # todo add metrics handling

    def train(
            self, train_dataloader: DataLoader, eval_dataloader: DataLoader = None, n_epochs: int = 1,
            train_epoch_losses: np.ndarray = None, eval_epoch_losses: np.ndarray = None,
            optimizer_state: list = None, verbose=0,
    ):
        eval_exists = eval_dataloader is not None
        train_epoch_losses = train_epoch_losses if train_epoch_losses is not None else np.empty(n_epochs)
        eval_epoch_losses = eval_epoch_losses if eval_epoch_losses is not None else np.empty(n_epochs)
        optimizer_state = optimizer_state if optimizer_state is not None else []
        mb_num = train_dataloader.get_batch_num()
        train_mb_losses = np.zeros(mb_num)

        # Initialize history
        self.history = History(n_epochs=n_epochs)
        self.history.before_training_cycle(self)

        # Metrics logs (to be passed to self.history)
        metric_logs = {metric.get_name(): None for metric in self.metrics}
        metric_logs['loss'] = None

        # Callbacks before training cycle
        train_dataloader.before_cycle()
        if eval_exists:
            eval_dataloader.before_cycle()

        for epoch in range(n_epochs):
            # First, set model to train mode
            self.set_to_train()
            # Callbacks before training epoch
            train_dataloader.before_epoch()
            self.optimizer.before_epoch()
            train_mb_losses.fill(0.)
            for mb in range(mb_num):

                mb_data = next(train_dataloader)
                if mb_data is None:
                    break
                input_mb, target_mb = mb_data[0], mb_data[1]
                y_hat = self.forward(input_mb)
                if isinstance(self.loss, RegularizedLoss):
                    data_loss_val, reg_loss_val = self.loss(y_hat, target_mb, layers=self.layers)
                else:
                    data_loss_val, reg_loss_val = self.loss(y_hat, target_mb)
                for metric in self.metrics:
                    metric.update(y_hat, target_mb)
                optim_log_dict = self.optimizer.to_log_dict()
                if verbose > 1:
                    print(
                        f'[Epoch {epoch}, Minibatch {mb}]{{',
                        f'\tloss values: (data = {data_loss_val.item():.8f}, '
                        f'regularization = {reg_loss_val.item():.8f})',
                        f'\toptimizer state: {optim_log_dict}',
                        f'}}',
                        sep='\n',
                    )
                train_mb_losses[mb] = np.mean(data_loss_val + reg_loss_val, axis=0).item()  # todo why this?
                # Backward of loss and hidden layers
                dvals = self.loss.backward(y_hat, target_mb)
                self.backward(dvals)
                self.optimizer.update(self.layers)
            train_dataloader.after_epoch()
            self.optimizer.after_epoch()
            train_epoch_losses[epoch] = np.mean(train_mb_losses).item()
            metric_logs['loss'] = train_epoch_losses[epoch]
            for metric in self.metrics:
                metric_logs[metric.get_name()] = metric.result()  # result at epoch level
                metric.reset()  # todo should we modify this for allowing multi-epochs metrics?
            self.history.after_training_epoch(self, epoch, logs=metric_logs)
            if verbose == 1:
                optim_log_dict = self.optimizer.to_log_dict()
                print(
                    f'[Epoch {epoch}]{{',
                    f'\tloss value: {train_epoch_losses[epoch]:.8f}',
                    f'\toptimizer state: {optim_log_dict}',
                    f'}}',
                    sep='\n',
                )

            # Set model to eval mode (useful e.g. for ModelCheckpoint callback independently from validation)
            self.set_to_eval()
            if eval_exists:
                eval_dataloader.before_epoch()
                input_eval, target_eval = next(eval_dataloader)
                y_hat = self.forward(input_eval)
                if isinstance(self.loss, RegularizedLoss):
                    data_loss_val, reg_loss_val = self.loss(y_hat, target_eval, layers=self.layers)
                else:
                    data_loss_val, reg_loss_val = self.loss(y_hat, target_eval)
                optim_log_dict = self.optimizer.to_log_dict()
                if verbose > 0:
                    print(
                        f'[Epoch {epoch}]{{',
                        f'\tloss values: (data = {data_loss_val.item():.8f}, '
                        f'regularization = {reg_loss_val.item():.8f})',
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

    # Utility method for more clariness when using model for predictions
    def predict(self, x: np.ndarray):
        self.set_to_eval()
        return self.forward(x)

    def reset(self):
        """
        Resets the model for a fresh usage.
        """
        self.optimizer = None
        self.loss = None
        self.metrics = None


__all__ = [
    'Model',
]
