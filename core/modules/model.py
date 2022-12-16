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

    @staticmethod
    def __objs_equal(self_obj, other_obj, include_all=False):
        """
        Utility method for equal() method.
        If include_all is True, returns True iff objects are
        equal, otherwise returns True iff they are both equal
        or the second one is None.
        """
        if not include_all and (self_obj is not None) and (other_obj is None):
            return True
        else:
            return self_obj == other_obj

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

    def save(self, file_path: str, include_compile_objs=True, include_history=True, serialize_all=False):
        """
        Saves model to a given file using pickle for serialization.
        :param file_path: Path of the file to which the model will be saved.
        :param include_compile_objs: If True, saves also optimizer, loss and training/validation metrics.
        :param include_history: If True, saves also current history.
        :param serialize_all: If True, saves layers with Layer._serialize_all enabled.
        """
        # Detach optimizer, loss and metrics if not requested
        loss, optimizer, metrics, validation_metrics, history = None, None, None, None, None
        if not include_compile_objs:
            optimizer = self.optimizer
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
        with open(file_path, 'wb') as fp:
            pickle.dump(self, fp)
        # Re-attach optimizer and loss if they were detached
        if not include_compile_objs:
            self.loss = loss
            self.optimizer = optimizer
            self.train_metrics = metrics
            self.validation_metrics = validation_metrics
        if not include_history:
            self.history = history

    def set_to_train(self):
        """
        Sets flags for indicating that the model is ready to train.
        """
        self.__is_training = True
        for layer in self.layers:
            layer.set_to_train()

    def set_to_eval(self, detach_history=False):
        """
        Resets flags that indicate that the model is ready to train
        for indicating that it is ready for validation / test.
        :param detach_history: If True, detaches self.history.
        """
        self.__is_training = False
        for layer in self.layers:
            layer.set_to_eval()
        if detach_history:
            history = self.history
            self.history = None
            return history
        return None

    def set_to_test(self):
        """
        Convenient alias for set_to_eval(detach_history=True).
        """
        return self.set_to_eval(detach_history=True)

    def is_training(self):
        return self.__is_training

    def forward(self, x: np.ndarray):
        current_output = x
        for i in range(self.length):
            layer = self.layers[i]
            current_output = layer.forward(current_output)
        return current_output

    def backward(self, delta_vals: np.ndarray):
        current_delta_vals = delta_vals
        for i in range(self.length):
            layer = self.layers[self.length - 1 - i]
            current_delta_vals = layer.backward(current_delta_vals)
        return current_delta_vals

    def compile(self, optimizer: Optimizer, loss: Loss, metrics: Metric | Sequence[Metric] = None):
        """
        Configures the model for training.
        """
        self.optimizer = optimizer
        # Auto-wrap loss into a regularization one
        # This is done because RegularizedLoss returns automatically the SAME data loss of the base
        # loss and the regularization term given by ALL the regularizers in the layers. Since
        # RegularizedLoss.forward() requires a set of layers from which to extract regularization
        # term, the only other solution is to scan model layers to check if there is any regularizer
        # in them and create a RegularizedLoss ONLY IN THAT CASE, but we think this solution is simpler
        # even if the regularizers are detected every time
        self.loss = RegularizedLoss(loss)
        metrics = [metrics] if isinstance(metrics, Metric) else metrics
        self.train_metrics = metrics if metrics is not None else []  # this ensures "None-safety" for the training loop
        self.validation_metrics = []
        self.stop_training = False

    @timeit
    def train(
            self, train_dataloader: DataLoader, eval_dataloader: DataLoader = None,
            max_epochs: int = 1, callbacks: Callback | Sequence[Callback] = None,
    ):
        """
        Main training (and validation) loop.
        :param train_dataloader: DataLoader for training data.
        :param eval_dataloader: DataLoader for validation data.
        :param max_epochs: Maximum number of epochs for training.
        :param callbacks: Callbacks to be passed for customizing behavior.
        :return: A `core.utils.History` object containing training
        and validation results.
        """
        validation_set_exists = eval_dataloader is not None
        callbacks = [] if callbacks is None else callbacks
        minibatch_number = train_dataloader.get_batch_number()
        train_minibatch_losses = np.zeros(minibatch_number)

        # Compile validation metrics if an eval dataset has been passed
        self.has_validation_set = validation_set_exists
        if self.has_validation_set:
            self._compile_validation_metrics(self.train_metrics)

        # Resets stop_training to False
        self.stop_training = False

        # Initialize history
        self.history = History(max_epochs=max_epochs)
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
        if validation_set_exists:
            eval_dataloader.before_cycle()
        for callback in callbacks:
            callback.before_training_cycle(self, logs=metric_logs)

        for epoch in range(max_epochs):
            if self.stop_training:
                break
            self.__train_epoch_loop(
                train_dataloader, epoch, max_epochs, metric_logs, callbacks, train_minibatch_losses, minibatch_number
            )
            self.__val_epoch_loop(epoch, metric_logs, callbacks, eval_dataloader)

        # Callbacks after training cycle
        train_dataloader.after_cycle()
        if validation_set_exists:
            eval_dataloader.after_cycle()
        for callback in callbacks:
            callback.after_training_cycle(self, logs=None)  # todo check also this!
        return self.history

    def _compile_validation_metrics(self, metrics: Metric | Sequence[Metric]):
        """
        Fills `self.validation_metrics` during training loop by using the given ones.
        """
        metrics = [metrics] if isinstance(metrics, Metric) else metrics
        for metric in metrics:
            val_metric = copy.deepcopy(metric)  # todo maybe a clone method is better
            val_metric.set_name(f'Val_{val_metric.get_name()}')
            self.validation_metrics.append(val_metric)

    def __train_epoch_loop(
            self, train_dataloader, epoch, max_epochs, metric_logs, callbacks, train_minibatch_losses, minibatch_number
    ):
        """
        Main training loop for a single epoch. Parameters have same names
        of the corresponding variables in `self.train()`.
        """
        # First, clean all old values for metric_logs and set model to train mode
        metric_logs['training'] = {k: None for k in metric_logs['training']}
        metric_logs['validation'] = {k: None for k in metric_logs['validation']}
        self.set_to_train()

        # Callbacks before training epoch
        train_dataloader.before_epoch()
        self.optimizer.before_epoch()
        for callback in callbacks:
            callback.before_training_epoch(self, epoch, logs=metric_logs)  # todo check if it is okay in general!
        train_minibatch_losses.fill(0.)

        # Minibatch-training loop
        for minibatch in range(minibatch_number):
            self.__train_minibatch_loop(
                train_dataloader, epoch, minibatch, metric_logs, callbacks, train_minibatch_losses
            )

        # Create and update metric logs for the elapsed epoch
        metric_logs['training']['loss'] = np.mean(train_minibatch_losses).item()
        for metric in self.train_metrics:
            result = metric.result()  # result at epoch level
            metric_logs['training'][metric.get_name()] = result
            metric.reset()  # todo should we modify this for allowing multi-epochs metrics?
        # Callbacks after training epoch
        train_dataloader.after_epoch()
        self.optimizer.after_epoch()
        # Sets stop_training to True before calling callbacks
        if epoch == max_epochs - 1:
            self.stop_training = True
        self.history.after_training_epoch(self, epoch, logs=metric_logs['training'])
        for callback in callbacks:
            callback.after_training_epoch(self, epoch, logs=metric_logs['training'])

    def __train_minibatch_loop(
            self, train_dataloader, epoch, minibatch, metric_logs, callbacks, train_minibatch_losses
    ):
        """
        Main training loop for a single minibatch. Parameters have same names
        of the corresponding variables in `self.train()`.
        """
        minibatch_data = next(train_dataloader)
        input_minibatch, target_minibatch = minibatch_data[0], minibatch_data[1]
        # Callbacks before training batch
        for callback in callbacks:
            callback.before_training_batch(self, epoch, minibatch, logs=metric_logs['training'])
        # (Training) metrics 'callback'
        for metric in self.train_metrics:
            metric.before_batch()
        net_output = self.forward(input_minibatch)
        if isinstance(self.loss, RegularizedLoss):
            data_loss_value, regularization_loss_value = self.loss(net_output, target_minibatch, layers=self.layers)
        else:
            data_loss_value, regularization_loss_value = self.loss(net_output, target_minibatch)
        # statement below is for not having any problem if loss reduction is None/'none'
        train_minibatch_losses[minibatch] = np.mean(data_loss_value + regularization_loss_value, axis=0).item()
        metric_logs['training']['loss'] = train_minibatch_losses[minibatch]
        # Backward of loss and hidden layers
        delta_vals = self.loss.backward(net_output, target_minibatch)
        self.backward(delta_vals)
        self.optimizer.update(self.layers)
        # Add output value to logs
        for metric in self.train_metrics:
            # Add metric reduction over current minibatch to logs
            metric_logs['training'][metric.get_name()] = metric.update(net_output, target_minibatch)
            metric.after_batch()
        # Callbacks after training batch
        for callback in callbacks:
            callback.after_training_batch(self, epoch, minibatch, logs=metric_logs['training'])

    def __val_epoch_loop(self, epoch, metric_logs, callbacks, eval_dataloader=None):
        """
        Main validation loop for a single epoch. Parameters have same names
        of the corresponding variables in `self.train()`.
        """
        validation_set_exists = eval_dataloader is not None
        # Set model to eval mode (useful e.g. for ModelCheckpoint callback independently from validation)
        self.set_to_eval()
        if validation_set_exists:
            # Callbacks before validation 'epoch'
            eval_dataloader.before_epoch()
            input_eval, target_eval = next(eval_dataloader)
            for callback in callbacks:
                callback.before_evaluate(self, epoch, logs=metric_logs['validation'])  # todo check this also!
            # Validation metrics 'callback'
            for validation_metric in self.validation_metrics:
                validation_metric.before_batch()
            net_output = self.forward(input_eval)
            if isinstance(self.loss, RegularizedLoss):
                data_loss_value, regularization_loss_value = self.loss(net_output, target_eval, layers=self.layers)
            else:
                data_loss_value, regularization_loss_value = self.loss(net_output, target_eval)
            eval_dataloader.after_epoch()
            metric_logs['validation']['Val_loss'] = np.mean(data_loss_value + regularization_loss_value, axis=0)
            for validation_metric in self.validation_metrics:
                metric_logs['validation'][validation_metric.get_name()] = \
                    validation_metric.update(net_output, target_eval)
                validation_metric.after_batch()
            # Callbacks after validation
            for callback in callbacks:
                callback.after_evaluate(self, epoch, logs=metric_logs['validation'])
            # Now update history with validation data
            self.history.after_evaluate(self, epoch, logs=metric_logs['validation'])

    def predict(self, x: np.ndarray):
        """
        Utility method for better readability when using model for predictions
        """
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
