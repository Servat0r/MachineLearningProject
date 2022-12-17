from itertools import product
from joblib import Parallel, delayed
import core.modules as cm
from core.utils.types import *
from core.utils.speedtests import timeit
from core.utils.initializers import RandomUniformInitializer, RandomNormalDefaultInitializer
from core.metrics import Metric
from core.callbacks import Callback, EarlyStopping, TrainingCSVLogger
import core.model_selection.validation as cv
from core.data import *
import numpy as np
import os
import json


# todo Need to do the following:
#  1. Optimize the part in which we maintain self.results: instead of saving ALL data,
#  maintain only the best (desired) ones (e.g. 4 in the example at the end);
#  2. Test with MONK and CUP data
#  3. Test with more parameter grids
#  4. Add a NON-Grid Search for searching DIRECTLY on a sequence of configurations (instead of cross-product)
#  (e.g. for the best models after the first coarse-grained grid search etc.)
#  5. Fix the bug of EarlyStopping with Val_loss (if it still exists)


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
        'size_hidden_layers': [(4,)],  # tuple of hidden units per layers

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
            ('es', 'Val_loss', 1e-3, 50),  # EarlyStopping with monitored metric, min_delta and patience
            ('l1', 1e-7),  # L1 with lambda
            ('l2', 1e-7),  # L2 with lambda
            ('l1l2', 1e-7, 1e-7)  # L1L2 with both lambdas
        ],

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
        layer_weight_initializer = self.convert_weights_initialization(config)

        # Weights and Biases regularizers (the same object can be used for all layers)
        layer_weights_regularizer, layer_biases_regularizer = self.convert_layer_regularization(config)

        layers = [cm.Input()]
        nr_hidden_layers = len(config['size_hidden_layers'])
        # input dimension unified with hidden sizes to simplify subsequent for loop
        non_output_sizes = (config['input_dim'],) + config['size_hidden_layers']
        for i in range(0, nr_hidden_layers):
            activation = self.convert_activation(config)
            layers.append(cm.Dense(
                non_output_sizes[i],
                non_output_sizes[i + 1],
                activation,
                weights_initializer=layer_weight_initializer,
                weights_regularizer=layer_weights_regularizer,
                biases_regularizer=layer_biases_regularizer,
            ))
        layers.append(cm.Linear(
            non_output_sizes[-1], config['output_dim'],
            weights_initializer=layer_weight_initializer,
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

    def convert_activation(self, data):
        activation_name = data.get('activation')
        if activation_name is None:
            raise ValueError(f"Activation function cannot be None")
        elif activation_name == 'tanh':
            activation = cm.Tanh()
        elif activation_name == 'sigmoid':
            activation = cm.Sigmoid()
        elif activation_name == 'relu':
            activation = cm.ReLU()
        else:
            raise TypeError(f'Unknown activation name {activation_name}')
        return activation

    def convert_decay(self, data):
        config = data.get('decay')
        start_lr_value = data.get('learning_rate')
        max_epochs = data.get('max_epoch')
        if (config is None) or (config == 'none'):
            return None
        scheduler_name, scheduler_args = config[0], config[1:]
        if scheduler_name == 'linear':
            # Multiply decay value by initial learning rate
            scheduler_args = (start_lr_value, start_lr_value * scheduler_args[0], max_epochs) + scheduler_args[1:]
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
        if config is None:
            raise ValueError(f"Weights initialization cannot be None!")
        elif config == 'Normal':
            return RandomNormalDefaultInitializer()
        elif config[0] == 'Uniform':
            return RandomUniformInitializer(*config[1:])
        else:
            raise TypeError(f"Unknown initialization strategy {config}")

    def convert_callbacks(self, data):
        callbacks = [TrainingCSVLogger(float_round_val=8)]
        reg_config = data.get('regularization')
        if reg_config is not None:
            reg_type, reg_vals = reg_config[0], reg_config[1:]
            if reg_type == 'es':
                callbacks.append(
                    EarlyStopping(
                        monitor=reg_vals[0], min_delta=reg_vals[1], patience=reg_vals[2],
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

    def __init__(self, parameters, scoring: Metric, cross_validator: cv.Validator = None):
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
            cv_random_state=None, epoch_shuffle=True, *args, **kwargs
    ):
        print(f'Using comb = {comb}')
        last_metric_values = []
        for train_data, eval_data in self.cross_validator.split(
                inputs, targets, shuffle=cv_shuffle, random_state=cv_random_state, *args, **kwargs
        ):
            model, optimizer, loss, callbacks = parameters_sequence.convert(comb)
            model.compile(optimizer, loss, metrics=[self.scoring_metric])
            train_dataset = ArrayDataset(*train_data)
            eval_dataset = ArrayDataset(*eval_data) if eval_data is not None else None

            train_dataloader = DataLoader(train_dataset, batch_size=comb['minibatch_size'], shuffle=epoch_shuffle)
            if eval_data is not None:
                eval_dataloader = DataLoader(eval_dataset, batch_size=len(eval_dataset))
            else:
                eval_dataloader = None
            history = model.train(
                train_dataloader, eval_dataloader, max_epochs=comb['max_epoch'], callbacks=callbacks
            )
            metric_values = history[f'Val_{self.scoring_metric.get_name()}']
            last_metric_value = metric_values[len(history) - 1]
            last_metric_values.append(last_metric_value)

        print(f'{len(last_metric_values)} last metric values for comb = {comb}')
        mean_metric_value = np.mean(last_metric_values)
        std_metric_value = np.std(last_metric_values)
        return {
            'config': comb,
            'mean': mean_metric_value.item(),
            'std': std_metric_value.item(),
        }

    @timeit
    def search(
            self, inputs: np.ndarray, targets: np.ndarray, cv_shuffle=True,
            cv_random_state=None, epoch_shuffle=True, n_jobs: int = os.cpu_count(),
            *args, **kwargs
    ):
        parameters_sequence = self.setup_parameters()
        hyperpar_comb = parameters_sequence.get_configs(self.parameters)
        if n_jobs is None:
            print('Doing Sequential Search')
            for comb in hyperpar_comb:
                self.results.append(self.__search_base_routine(
                    parameters_sequence, comb, inputs, targets, cv_shuffle,
                    cv_random_state, epoch_shuffle, *args, **kwargs
                ))
        else:
            print(f'Doing Parallel Search with {n_jobs} workers')
            self.results = Parallel(n_jobs)(
                delayed(BaseSearch.__search_base_routine)(
                    self, parameters_sequence, comb, inputs, targets, cv_shuffle,
                    cv_random_state, epoch_shuffle, *args, **kwargs
                ) for comb in hyperpar_comb
            )
        self.results = sorted(self.results, key=lambda x: x['mean'])

    def save_best(self, number: int, directory_path: str = '.', file_name: str = 'best_results.json'):
        results = self.results[:number]
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, 'w') as fp:
            json.dump(results, fp, indent=2)

    def save_all(self, directory_path: str = '.', file_name: str = 'results.json'):
        number = len(self.results)
        self.save_best(number, directory_path, file_name)


class GridSearch(BaseSearch):

    def setup_parameters(self) -> ParameterSequence:
        return ParameterGrid()


class FixedCombSearch(BaseSearch):

    def setup_parameters(self) -> ParameterSequence:
        return ParameterList()


__all__ = [
    'ParameterSequence', 'ParameterGrid', 'ParameterList',
    'BaseSearch', 'GridSearch', 'FixedCombSearch',
]
