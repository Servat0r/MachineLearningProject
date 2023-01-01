from itertools import product
from joblib import Parallel, delayed
import core.modules as cm
from core.utils.types import *
from core.utils.initializers import *
from core.metrics import Metric
from core.callbacks import Callback, EarlyStopping, TestSetMonitor
import core.model_selection.validation as cv
from core.data import *
import numpy as np
import os
import json
from time import perf_counter
from copy import deepcopy


def cross_product(inp: dict):
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))


class ParameterSequence:
    """
    Base class representing a sequence of parameters for a model-selection
    search: accepts a well-formatted input describing configurations for
    the models to be tested and returns each time the corresponding model,
    optimizer, loss etc.

    Syntax is as following:
    {
        'size_hidden_layers': [[4]],  # tuple of hidden units per layers

        'input_dim': [10],  # Input dimension

        'output_dim': [2],  # Output dimension

        'activation': ['tanh'],  # Activation function for all the layers: 'tanh' / 'sigmoid' / 'relu'

        'learning_rate': [1e-3, 1e-4],  # (Initial) learning rate

        'decay': [
            'none',  # No weight decay
            ('linear', end_value / start_value, round_val),  # Linear decay
            ('iter', decay, round_val),  # Iter-based decay
            ('exponential', alpha, round_val)  # Exponential decay
        ]

        'momentum': [0.0, 0.1],  # momentum values

        'regularization': [
            'none',  # No regularization
            ('l1', 1e-7),  # L1 with lambda
            ('l2', 1e-7),  # L2 with lambda
            ('l1l2', 1e-7, 1e-7)  # L1L2 with both lambdas
        ],

        'early_stopping': [
            'none',  # No ES
            ('Val_loss', 1e-3, 50),  # EarlyStopping with monitored metric, min_delta and patience
        ]

        'weights_initialization': [
            'Normal',  # Normal (mean = 0, std = 1) distribution
            ('Uniform', -0.1, 0.1)  # Uniform distribution with min and max values
        ],

        'max_epoch': [250],  # Maximum number of epochs

        'minibatch_size': [4]  # Minibatch size

    }
    """
    @abstractmethod
    def get_configs(self, data) -> Generator[dict, Any, None]:
        pass

    def convert(self, config: dict) -> tuple[cm.Model, cm.Optimizer, cm.Loss, list[Callback]]:
        # Weights initializers
        layer_weight_initializers = self.convert_weights_initialization(config)

        # Weights and Biases regularizers (the same object can be used for all layers)
        layer_weights_regularizer, layer_biases_regularizer = self.convert_layer_regularization(config)

        layers = [cm.Input()]
        nr_hidden_layers = len(config['size_hidden_layers'])
        # input dimension unified with hidden sizes to simplify subsequent for loop
        non_output_sizes = [config['input_dim']] + config['size_hidden_layers']
        activations = self.convert_activation(config)
        for i in range(0, nr_hidden_layers):
            layers.append(cm.Dense(
                non_output_sizes[i],
                non_output_sizes[i + 1],
                activations[i],
                weights_initializer=layer_weight_initializers[i],
                weights_regularizer=layer_weights_regularizer,
                biases_regularizer=layer_biases_regularizer,
            ))
        layers.append(cm.Linear(
            non_output_sizes[-1], config['output_dim'],
            weights_initializer=layer_weight_initializers[-1],
            weights_regularizer=layer_weights_regularizer,
            biases_regularizer=layer_biases_regularizer,
        ))
        model = cm.Model(layers)
        scheduler = self.convert_decay(config)
        optimizer = cm.SGD(
            config['learning_rate'], lr_decay_scheduler=scheduler, momentum=config['momentum']
        )
        loss = cm.MSELoss(const=1.0, reduction='mean')
        callbacks = self.convert_callbacks(config)
        return model, optimizer, loss, callbacks

    def __get_act_from_string(self, act_string: str):
        if act_string == 'tanh':
            return cm.Tanh()
        elif act_string == 'sigmoid':
            return cm.Sigmoid()
        elif act_string == 'relu':
            return cm.ReLU()
        else:
            return None

    def convert_activation(self, data):
        activation_data = data.get('activation')
        nr_hidden_layers = len(data.get('size_hidden_layers'))
        activations = []
        if activation_data is None:
            raise ValueError(f"Activation function cannot be None")
        elif isinstance(activation_data, str):
            for i in range(nr_hidden_layers):
                activations.append(self.__get_act_from_string(activation_data))
        elif isinstance(activations, Sequence):
            for i in range(nr_hidden_layers):
                activations.append(self.__get_act_from_string(activation_data[i]))
        else:
            raise TypeError(f'Unknown activation name {activation_data}')
        return activations

    def convert_decay(self, data):
        config = data.get('decay')
        start_lr_value = data.get('learning_rate')
        max_epochs = data.get('max_epoch')
        if (config is None) or (config == 'none'):
            return None
        scheduler_name, scheduler_args = config[0], config[1:]
        if scheduler_name == 'linear':
            # Multiply decay value by initial learning rate
            scheduler_args = (start_lr_value, start_lr_value * scheduler_args[0],
                              max_epochs) + tuple(scheduler_args[1:])
            return cm.LinearDecayScheduler(*scheduler_args)
        elif scheduler_name == 'iter':
            return cm.IterBasedDecayScheduler(start_lr_value, *scheduler_args)
        elif scheduler_name == 'exponential':
            return cm.ExponentialDecayScheduler(start_lr_value, *scheduler_args)
        else:
            raise TypeError(f"Unkwown scheduler config {config}")

    def convert_layer_regularization(self, data):
        layer_weights_regularizer = None
        layer_biases_regularizer = None
        config = data.get('regularization')
        if (config is not None) and (config != 'none'):
            reg_type, reg_vals = config[0], config[1:]
            if reg_type == 'l1':
                layer_weights_regularizer = cm.L1Regularizer(l1_lambda=reg_vals[0])
                layer_biases_regularizer = cm.L1Regularizer(l1_lambda=reg_vals[0])
            elif reg_type == 'l2':
                layer_weights_regularizer = cm.L2Regularizer(l2_lambda=reg_vals[0])
                layer_biases_regularizer = cm.L2Regularizer(l2_lambda=reg_vals[0])
            elif reg_type == 'l1l2':
                layer_weights_regularizer = cm.L1L2Regularizer(l1_lambda=reg_vals[0], l2_lambda=reg_vals[1])
                layer_biases_regularizer = cm.L1L2Regularizer(l1_lambda=reg_vals[0], l2_lambda=reg_vals[1])
            # Other cases are EarlyStopping or NO regularization
        return layer_weights_regularizer, layer_biases_regularizer

    def convert_weights_initialization(self, data):
        config = data.get('weights_initialization')
        nr_hidden_layers = len(data['size_hidden_layers']) + 1
        if config is None:
            raise ValueError(f"Weights initialization cannot be None!")
        elif config[0] == 'Normal':
            return nr_hidden_layers * [RandomNormalDefaultInitializer(*config[1:])]  # scale, seed
        elif config[0] == 'FanOut':
            sizes = data['size_hidden_layers'] + [data['output_dim']]
            inits = []
            for i in range(nr_hidden_layers):
                inits.append(FanInitializer(sizes[i], seed=config[1]))
            return inits
        elif config[0] == 'Uniform':
            return nr_hidden_layers * [RandomUniformInitializer(*config[1:])]
        else:
            raise TypeError(f"Unknown initialization strategy {config}")

    def convert_callbacks(self, data):
        monitor = data.get('monitor')
        # if monitor is None:
        #     raise ValueError(f"Monitored metric cannot be None!")
        # callbacks = [ModelMonitor(monitor, mode='min', return_best_result=True)]
        callbacks = []
        reg_config = data.get('early_stopping')
        if (reg_config is not None) and (reg_config != 'none'):
            callbacks.append(
                EarlyStopping(
                    monitor=monitor, min_delta=reg_config[0], patience=reg_config[1],
                    mode='min', return_best_result=False,
                )
            )
        return callbacks


class ParameterGrid(ParameterSequence):
    """
    A grid of parameters s.t. each possible combination of each value for
    each parameter is a candidate configuration.
    """
    def get_configs(self, data) -> Generator[dict, Any, None]:
        hyperpar_comb = cross_product(data)
        for comb in hyperpar_comb:
            yield comb


class ParameterList(ParameterSequence):
    """
    A list of "already-compiled" configurations to be tested all.
    """
    def get_configs(self, data) -> Generator[dict, Any, None]:
        for config in data:
            yield config


class BaseSearch:
    """
    Base class representing a search over a hyperparameter(s) space.
    """

    def __init__(self, parameters, scoring: Metric, cross_validator: cv.Validator = None):
        """
        :param parameters: An object (dict, list, ..., depending on the specific class)
        that is used for generating all the needed configurations.
        :param scoring: A metric used to evaluate each configuration performance and sort
        all them at the end.
        :param cross_validator: Validator to be used for splitting and iterating over
        training and validation data, e.g. Holdout or KFold. Defaults to None.
        """
        self.parameters = parameters
        self.scoring_metric = scoring
        self.cross_validator = cross_validator if cross_validator is not None else cv.Holdout()
        self._cv_not_None = cross_validator is not None
        self.results = []  # results of each grid search case

    @abstractmethod
    def setup_parameters(self) -> ParameterSequence:
        pass

    def __search_base_routine(
            self, parameters_sequence: ParameterSequence, comb: dict,
            inputs: np.ndarray, targets: np.ndarray, cv_shuffle=True,
            cv_random_state=None, epoch_shuffle=True, test_set_data: np.ndarray = None,
            test_set_targets: np.ndarray = None, *args, **kwargs
    ):
        """
        Main loop of the search.
        """
        print(f'Using comb = {comb}')
        best_metric_values = []
        test_set_values = []
        for train_data, eval_data in self.cross_validator.split(
                inputs, targets, shuffle=cv_shuffle, random_state=cv_random_state, *args, **kwargs
        ):
            model, optimizer, loss, callbacks = parameters_sequence.convert(comb)
            model.compile(optimizer, loss, metrics=[self.scoring_metric])
            train_dataset = ArrayDataset(*train_data)
            eval_dataset = ArrayDataset(*eval_data) if eval_data is not None else None

            if (test_set_data is not None) and (test_set_targets is not None):
                test_set_monitor = TestSetMonitor(
                    test_set_data, test_set_targets, [deepcopy(self.scoring_metric)],
                    comb['max_epoch'], raw_outputs=False
                )
            else:
                test_set_monitor = None

            train_dataloader = DataLoader(train_dataset, batch_size=comb['minibatch_size'], shuffle=epoch_shuffle)
            if eval_data is not None:
                eval_dataloader = DataLoader(eval_dataset, batch_size=len(eval_dataset))
            else:
                eval_dataloader = None
            history = model.train(
                train_dataloader, eval_dataloader, max_epochs=comb['max_epoch'], callbacks=callbacks
            )
            metric_values = history[f'Val_{self.scoring_metric.get_name()}']
            last_metric_value = metric_values[len(history) - 1].item()
            best_metric_values.append(last_metric_value)
            if test_set_monitor is not None:
                test_set_values.append(test_set_monitor[self.scoring_metric.get_name()][-1].item())

        mean_metric_value = np.mean(best_metric_values)
        std_metric_value = np.std(best_metric_values)
        return {
            'config': comb,
            'mean': mean_metric_value.item(),
            'std': std_metric_value.item(),
            'values': best_metric_values,
            'test_values': test_set_values,
        }

    def search(
            self, inputs: np.ndarray, targets: np.ndarray, cv_shuffle=True,
            cv_random_state=None, epoch_shuffle=True, n_jobs: int = os.cpu_count(),
            search_stats_file='search.txt', test_set_data: np.ndarray = None,
            test_set_targets: np.ndarray = None, *args, **kwargs
    ):
        """
        Main searching method.
        :param inputs: Input data (development set).
        :param targets: Input targets (development set).
        :param cv_shuffle: Passed to the underlying cross validator
        split() method each time.
        :param cv_random_state: Passed to the underlying cross validator
        split() method each time.
        :param epoch_shuffle: Passed to the training DataLoaders for each
        training cycle.
        :param n_jobs: If None, performs the search sequentially. Otherwise,
        n_jobs workers (subprocesses) are used.
        :param search_stats_file: Path of the file in which to write common
        statistics about the search. Defaults to 'search.txt'.
        :param args: Extra positional arguments to be passed to the cross
        validator.
        :param kwargs: Extra keyword arguments to be passed to the cross
        validator.
        """
        parameters_sequence = self.setup_parameters()
        hyperpar_comb = parameters_sequence.get_configs(self.parameters)
        current_time = perf_counter()
        if n_jobs is None:
            print('Doing Sequential Search')
            for comb in hyperpar_comb:
                self.results.append(self.__search_base_routine(
                    parameters_sequence, comb, inputs, targets, cv_shuffle,
                    cv_random_state, epoch_shuffle, test_set_data,
                    test_set_targets, *args, **kwargs
                ))
        else:
            print(f'Doing Parallel Search with {n_jobs} workers')
            self.results = Parallel(n_jobs)(
                delayed(BaseSearch.__search_base_routine)(
                    self, parameters_sequence, comb, inputs, targets, cv_shuffle,
                    cv_random_state, epoch_shuffle, *args, **kwargs
                ) for comb in hyperpar_comb
            )
        current_time = perf_counter() - current_time
        with open(search_stats_file, 'w') as fp:
            print(f"Elapsed time for search: {current_time}", file=fp)
            print(f"Number of workers used: {n_jobs}", file=fp)
        self.results = sorted(self.results, key=lambda x: x['mean'])

    def save_best(
            self, number: int, directory_path: str = '.',
            file_name: str = 'best_results.json'
    ):
        results = self.results[:number]
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, 'w') as fp:
            json.dump(results, fp, indent=2)

    def save_all(self, directory_path: str = '.', file_name: str = 'results.json'):
        number = len(self.results)
        self.save_best(number, directory_path, file_name)


class GridSearch(BaseSearch):
    """
    Grid Search: __init__ method accepts a dict of the form {str:list}
    as first parameter and cycles over all the possible combinations.
    """

    def setup_parameters(self) -> ParameterSequence:
        return ParameterGrid()


class FixedCombSearch(BaseSearch):
    """
    A search performed on all the configurations in a list of them
    given as first parameter in __init__.
    """

    def setup_parameters(self) -> ParameterSequence:
        return ParameterList()


__all__ = [
    'ParameterSequence', 'ParameterGrid', 'ParameterList',
    'BaseSearch', 'GridSearch', 'FixedCombSearch',
]
