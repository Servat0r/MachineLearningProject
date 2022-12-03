# Model class: main class for building a complete Neural Network
from __future__ import annotations
import pickle
import copy

from core.utils import *
from core.data import *
from .layers import *
from .losses import *
from .optimizers import *
from ..metrics import *
from ..callbacks import *


class Model:
    """
    Base class for a Neural Network
    """
    @property
    def metrics(self):
        train_metrics = [] if self.train_metrics is None else self.train_metrics
        validation_metrics = [] if self.validation_metrics is None else self.validation_metrics
        return train_metrics + validation_metrics
    
    def __init__(self, layers: Layer | Sequence[Layer]):
        self.layers: Sequence[Layer] = [layers] if isinstance(layers, Layer) else list(layers)
        self.optimizer = None
        self.loss = None
        self.train_metrics = None
        self.validation_metrics = None
        self.length = len(self.layers)
        self.history = None  # will be filled during training
        self.__is_training = None
        self.has_validation_set = None
        self.stop_training = None

    @staticmethod
    def __objs_equal(self_obj, other_obj, include_all=False):
        if self_obj is not None and other_obj is not None:
            return self_obj == other_obj
        elif include_all:
            return self_obj == other_obj  # both None
        return True

    def equal(self, other: Model, include_updates=False, include_all=False):
        # First check layers
        for layer1, layer2 in zip(self.layers, other.layers):
            if not layer1.equals(layer2, include_updates, include_all):
                return False
        # Then, check compile objects
        check = [
            self.__objs_equal(self.optimizer, other.optimizer, include_all),
            self.__objs_equal(self.loss, other.loss, include_all),
            self.__objs_equal(self.train_metrics, other.train_metrics, include_all),
            self.__objs_equal(self.validation_metrics, other.validation_metrics, include_all),
            self.__objs_equal(self.length, other.length, include_all),
            self.__objs_equal(self.has_validation_set, other.has_validation_set, include_all),
        ]
        if not all(check):
            return False
        if (self.history is not None) and (other.history is not None):
            if len(self.history) == len(other.history) and self.history != other.history:
                return False
        return True

    def __eq__(self, other):
        return isinstance(other, Model) and self.equal(other, include_all=True)

    def get_parameters(self, copy=False) -> list[dict]:
        return [layer.get_parameters(copy=copy) for layer in self.layers]

    def set_parameters(self, params: list):
        for layer, param_dict in zip(self.layers, params):
            layer.set_parameters(param_dict)

    def save_parameters(self, fpath: str):
        with open(fpath, 'wb') as fp:
            pickle.dump(self.get_parameters(), fp)  # todo if we introduce multithreading, copy -> True

    def load_parameters(self, fpath: str):
        with open(fpath, 'rb') as fp:
            self.set_parameters(pickle.load(fp))

    @staticmethod
    def load(fpath: str) -> Model:
        with open(fpath, 'rb') as fp:
            model = pickle.load(fp)
        return model

    def save(self, fpath: str, include_compile_objs=True, include_history=True, serialize_all=False):
        # Detach optimizer, loss and metrics if not requested
        loss, optim, metrics, validation_metrics, history = None, None, None, None, None
        if not include_compile_objs:
            optim = self.optimizer
            loss = self.loss
            metrics = self.train_metrics
            validation_metrics = self.validation_metrics
            self.optimizer = None
            self.loss = None
            self.train_metrics = None
            self.validation_metrics = None
        if not include_history:
            history = self.history
            self.history = None
        # Set/Unset _serialize_all flag for layers if requested
        for layer in self.layers:
            if serialize_all:
                layer.set_serialize_all()
            else:
                layer.unset_serialize_all()
        # Serialize model
        with open(fpath, 'wb') as fp:
            pickle.dump(self, fp)
        # Re-attach optimizer and loss if they were detached
        if not include_compile_objs:
            self.loss = loss
            self.optimizer = optim
            self.train_metrics = metrics
            self.validation_metrics = validation_metrics
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

    def set_to_test(self):
        # Convenient alias for set_to_eval(detach_history=True)
        return self.set_to_eval(detach_history=True)

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

    # TODO Modify in order to have train-only and validation-only losses!
    def compile(self, optimizer: Optimizer, loss: Loss, metrics: Metric | Sequence[Metric] = None):
        """
        Configures the model for training.
        """
        # todo create metrics (accuracy, mse/abs/mee, timing, ram usage)!
        self.optimizer = optimizer
        # auto-wrap loss into a regularization one
        self.loss = RegularizedLoss(loss)
        metrics = [metrics] if isinstance(metrics, Metric) else metrics
        self.train_metrics = metrics if metrics is not None else []  # this ensures "None-safety" for the training loop
        self.validation_metrics = []
        self.stop_training = False

    def _compile_validation_metrics(self, metrics: Metric | Sequence[Metric]):
        metrics = [metrics] if isinstance(metrics, Metric) else metrics
        for metric in metrics:
            val_metric = copy.deepcopy(metric)  # todo maybe a clone method is better
            val_metric.set_name(f'Val_{val_metric.get_name()}')
            self.validation_metrics.append(val_metric)

    def add_metric(self, metric: Metric):
        self.train_metrics.append(metric)
        if self.has_validation_set:
            self._compile_validation_metrics(metric)

    def __train_epoch_loop(self, train_dataloader, epoch, n_epochs, metric_logs, callbacks, train_mb_losses, mb_num):
        # First, clean all old values for metric_logs and set model to train mode
        metric_logs['training'] = {k: None for k in metric_logs['training']}
        metric_logs['validation'] = {k: None for k in metric_logs['validation']}
        self.set_to_train()

        # Callbacks before training epoch
        train_dataloader.before_epoch()
        self.optimizer.before_epoch()
        for callback in callbacks:
            callback.before_training_epoch(self, epoch, logs=metric_logs)  # todo check if it is okay in general!
        train_mb_losses.fill(0.)

        # Minibatch-training loop
        for mb in range(mb_num):
            self.__train_mb_loop(train_dataloader, epoch, mb, metric_logs, callbacks, train_mb_losses)

        # Create and update metric logs for the elapsed epoch
        metric_logs['training']['loss'] = np.mean(train_mb_losses).item()
        for metric in self.train_metrics:
            result = metric.result()  # result at epoch level
            metric_logs['training'][metric.get_name()] = result
            metric.reset()  # todo should we modify this for allowing multi-epochs metrics?
        # Callbacks after training epoch
        train_dataloader.after_epoch()
        self.optimizer.after_epoch()
        # Sets stop_training to True before calling callbacks
        if epoch == n_epochs - 1:
            self.stop_training = True
        self.history.after_training_epoch(self, epoch, logs=metric_logs['training'])
        for callback in callbacks:
            callback.after_training_epoch(self, epoch, logs=metric_logs['training'])

    def __train_mb_loop(self, train_dataloader, epoch, mb, metric_logs, callbacks, train_mb_losses):
        mb_data = next(train_dataloader)
        input_mb, target_mb = mb_data[0], mb_data[1]
        # Callbacks before training batch
        for callback in callbacks:
            callback.before_training_batch(self, epoch, mb, logs=metric_logs['training'])
        # (Training) metrics 'callback'
        for metric in self.train_metrics:
            metric.before_batch()
        y_hat = self.forward(input_mb)
        if isinstance(self.loss, RegularizedLoss):
            data_loss_val, reg_loss_val = self.loss(y_hat, target_mb, layers=self.layers)
        else:
            data_loss_val, reg_loss_val = self.loss(y_hat, target_mb)
        # statement below is for not having any problem if loss reduction is None/'none'
        train_mb_losses[mb] = np.mean(data_loss_val + reg_loss_val, axis=0).item()
        metric_logs['training']['loss'] = train_mb_losses[mb]
        # Backward of loss and hidden layers
        dvals = self.loss.backward(y_hat, target_mb)
        self.backward(dvals)
        self.optimizer.update(self.layers)
        # Add output value to logs
        for metric in self.train_metrics:
            # Add metric reduction over current minibatch to logs
            metric_logs['training'][metric.get_name()] = metric.update(y_hat, target_mb)
            metric.after_batch()
        # Callbacks after training batch
        for callback in callbacks:
            callback.after_training_batch(self, epoch, mb, logs=metric_logs['training'])

    def __val_epoch_loop(self, epoch, metric_logs, callbacks, eval_dataloader=None):
        eval_exists = eval_dataloader is not None
        # Set model to eval mode (useful e.g. for ModelCheckpoint callback independently from validation)
        self.set_to_eval()
        if eval_exists:
            # Callbacks before validation 'epoch'
            eval_dataloader.before_epoch()
            input_eval, target_eval = next(eval_dataloader)
            for callback in callbacks:
                callback.before_evaluate(self, epoch, logs=metric_logs['validation'])  # todo check this also!
            # Validation metrics 'callback'
            for val_metric in self.validation_metrics:
                val_metric.before_batch()
            y_hat = self.forward(input_eval)
            if isinstance(self.loss, RegularizedLoss):
                data_loss_val, reg_loss_val = self.loss(y_hat, target_eval, layers=self.layers)
            else:
                data_loss_val, reg_loss_val = self.loss(y_hat, target_eval)
            eval_dataloader.after_epoch()
            metric_logs['validation']['Val_loss'] = np.mean(data_loss_val + reg_loss_val, axis=0)
            for val_metric in self.validation_metrics:
                metric_logs['validation'][val_metric.get_name()] = val_metric.update(y_hat, target_eval)
                val_metric.after_batch()
            # Callbacks after validation
            for callback in callbacks:
                callback.after_evaluate(self, epoch, logs=metric_logs['validation'])
            # Now update history with validation data
            self.history.after_evaluate(self, epoch, logs=metric_logs['validation'])

    def train(
            self, train_dataloader: DataLoader, eval_dataloader: DataLoader = None,
            n_epochs: int = 1, callbacks: Callback | Sequence[Callback] = None,
    ):
        eval_exists = eval_dataloader is not None
        callbacks = [] if callbacks is None else callbacks
        mb_num = train_dataloader.get_batch_num()
        train_mb_losses = np.zeros(mb_num)

        # Compile validation metrics if an eval dataset has been passed
        self.has_validation_set = eval_exists
        if self.has_validation_set:
            self._compile_validation_metrics(self.train_metrics)

        # Resets stop_training to False
        self.stop_training = False

        # Initialize history
        self.history = History(n_epochs=n_epochs)
        self.history.before_training_cycle(self)

        # Metrics logs (to be passed to self.history)
        metric_logs = {
            'training': {'loss': None},
            'validation': {},
        }
        metric_logs['training'].update({metric.get_name(): None for metric in self.train_metrics})

        # Add validation metrics
        if self.has_validation_set:
            metric_logs['validation']['Val_loss'] = None
            metric_logs['validation'].update({val_metric.get_name(): None for val_metric in self.validation_metrics})

        # Callbacks before training cycle
        train_dataloader.before_cycle()
        if eval_exists:
            eval_dataloader.before_cycle()
        for callback in callbacks:
            callback.before_training_cycle(self, logs=metric_logs)

        for epoch in range(n_epochs):
            if self.stop_training:
                break
            self.__train_epoch_loop(train_dataloader, epoch, n_epochs, metric_logs, callbacks, train_mb_losses, mb_num)
            self.__val_epoch_loop(epoch, metric_logs, callbacks, eval_dataloader)

        # Callbacks after training cycle
        train_dataloader.after_cycle()
        if eval_exists:
            eval_dataloader.after_cycle()
        for callback in callbacks:
            callback.after_training_cycle(self, logs=None)  # todo check also this!
        return self.history

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
        self.train_metrics = []
        self.validation_metrics = []


__all__ = [
    'Model',
]
