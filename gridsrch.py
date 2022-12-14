from itertools import product
import core.modules as cm
from core.utils.types import *
from core.utils import RandomUniformInitializer, RandomNormalDefaultInitializer
from core.metrics import Metric, MEE
from core.callbacks import EarlyStopping, TrainingCSVLogger
import core.validation as cv
from core.data import *
import numpy as np
import os
import json
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


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


class GridSearch:

    def __init__(self, param_grid: dict[str, Any], scoring: Metric, cross_validator: cv.Validator = None):
        self.param_grid = param_grid
        self.scoring_metric = scoring
        self.cross_validator = cross_validator if cross_validator is not None else cv.Holdout()
        self._cv_not_None = cross_validator is not None
        self.results = []  # results of each grid search case

    def search(
            self, inputs: np.ndarray, targets: np.ndarray, cv_shuffle=True,
            cv_random_state=None, epoch_shuffle=True, *args, **kwargs
    ):
        hyperpar_comb = cross_product(self.param_grid)
        for comb in hyperpar_comb:

            # Create model, optimizer, loss etc  # todo refactor in a separate method!
            # Weights initializers
            if comb['weight_distr'] == 'Normal':
                layer_weight_initializer = RandomNormalDefaultInitializer()
            else:
                layer_weight_initializer = RandomUniformInitializer(*comb['weight_distr'])

            # Weights and Biases regularizers (the same object can be used for all layers)
            reg_type, reg_vals = comb['regul'][0], comb['regul'][1:]
            if reg_type == 'l1':
                layer_weights_regularizer = cm.L1Regularizer(l1_lambda=reg_vals[0])
                layer_biases_regularizer = cm.L1Regularizer(l1_lambda=reg_vals[0])
            elif reg_type == 'l2':
                layer_weights_regularizer = cm.L2Regularizer(l2_lambda=reg_vals[0])
                layer_biases_regularizer = cm.L2Regularizer(l2_lambda=reg_vals[0])
            elif reg_type == 'l1l2':
                layer_weights_regularizer = cm.L1L2Regularizer(l1_lambda=reg_vals[0], l2_lambda=reg_vals[1])
                layer_biases_regularizer = cm.L1L2Regularizer(l1_lambda=reg_vals[0], l2_lambda=reg_vals[1])
            else:  # EarlyStopping or anything else
                layer_weights_regularizer = None
                layer_biases_regularizer = None

            layers = [cm.Input()]
            nr_hidden_layers = len(comb['size_hidden_layers'])
            # input dimension unified with hidden sizes to simplify subsequent for loop
            non_output_sizes = (comb['input_dim'],) + comb['size_hidden_layers']
            for i in range(0, nr_hidden_layers):
                activation_name = comb['activation_function']
                if activation_name == 'tanh':
                    activation = cm.Tanh()
                elif activation_name == 'sigmoid':
                    activation = cm.Sigmoid()
                else:
                    activation = cm.ReLU()
                layers.append(cm.Dense(
                    non_output_sizes[i],
                    non_output_sizes[i + 1],
                    activation,
                    weights_initializer=layer_weight_initializer,
                    weights_regularizer=layer_weights_regularizer,
                    biases_regularizer=layer_biases_regularizer,
                ))
            layers.append(cm.Linear(
                non_output_sizes[-1], comb['output_dim'],
                weights_initializer=layer_weight_initializer,
                weights_regularizer=layer_weights_regularizer,
                biases_regularizer=layer_biases_regularizer,
            ))
            # da qua in poi definisco il modello
            model = cm.Model(layers)
            # todo il decay del learning rate lo devi fare passando un oggetto core.modules.schedulers.Scheduler con
            #  opportuni parametri per __init__
            optimizer = cm.SGD(comb['learning_rate'], momentum=comb['momentum'])
            loss = cm.MSELoss(const=1., reduction='mean')
            model.compile(optimizer, loss, metrics=[self.scoring_metric])
            if reg_type == 'es':
                callbacks = [
                    EarlyStopping(
                        monitor=f'Val_{self.scoring_metric.get_name()}', min_delta=reg_vals[0], patience=comb['patience'],
                        mode='min', return_best_result=True  # todo Val_loss
                    ),
                ]
            else:
                callbacks = []
            callbacks.append(TrainingCSVLogger(float_round_val=8))

            last_metric_values = []
            for train_data, eval_data in self.cross_validator.split(
                    inputs, targets, shuffle=cv_shuffle, random_state=cv_random_state, *args, **kwargs
            ):
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
                last_metric_value = metric_values[len(history)-1]
                last_metric_values.append(last_metric_value)

            mean_metric_value = np.mean(last_metric_values)
            std_metric_value = np.std(last_metric_values)
            self.results.append({
                'config': comb,
                'mean': mean_metric_value.item(),
                'std': std_metric_value.item(),
            })
            self.results = sorted(self.results, key=lambda x: x['mean'])

    def save_best(self, number: int, directory_path: str = '.', file_name: str = 'best_results.json'):
        results = self.results[:number]
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, 'w') as fp:
            json.dump(results, fp, indent=2)


example_param_of_search = {
    'size_hidden_layers': [(4,)],
    'input_dim': [10],
    'output_dim': [2],
    'activation_function': ['tanh'],
    'learning_rate': [1e-3, 1e-4],
    'momentum': [0.0, 0.1],
    'regul': [('es', 1e-3)],  # [('l1', 1e-7), ('l2', 1e-7), ('es', 1e-3)],
    'weight_distr': [(-0.1, 0.1)],
    'patience': [50],
    'max_epoch': [250],
    'minibatch_size': [4]
}

X, y = load_diabetes(return_X_y=True)
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

X_dev = np.expand_dims(X_dev, axis=1)
X_test = np.expand_dims(X_test, axis=1)
y_dev = np.reshape(y_dev, (y_dev.shape[0], 1, 1))
y_test = np.reshape(y_test, (y_test.shape[0], 1, 1))

# this is for using k-fold
k_fold_grid_search = GridSearch(example_param_of_search, MEE(), cv.KFold(6))
k_fold_grid_search.search(X_dev, y_dev, cv_shuffle=True, cv_random_state=0, epoch_shuffle=True)
# this instead is for holdout
holdout_grid_search = GridSearch(example_param_of_search, MEE(), cv.Holdout())
holdout_grid_search.search(
    X_dev, y_dev, cv_shuffle=True, cv_random_state=0, epoch_shuffle=True, validation_split_percentage=0.2
)

# Save the best 4 models
k_fold_grid_search.save_best(4)

param_of_search = {
    'size_hidden_layers': [(4, 2), (1, 1)],
    'input_dim': [9],
    'output_dim': [2],
    'activation_function': ['tanh', 'sigmoid'],
    'learning_rate': [1e-2, 1e-3, 1e-4, 1e-5],
    'decay': [1],  # todo che decay stai usando?
    # metterei nella lista direttamente gli oggetti con i parametri (non ho idee migliori)
    'momentum': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    # questo dice se regolarizzazione o early stopping e con quali parametri
    'regul': [('l1', 1), ('l2', 1), ('l1l2', 1, 1), ('es', 1)],
    # quando esce 'Normal' inizializzo con la normal, altrimenti con la uniform con gli estremi che mi escono
    'weight_distr': [(0, 1), 'Normal'],
    'patience': [0, 10, 20, 30, 40],
    'max_epoch': [250, 500, 750, 1000],
    'size_minibatch': [4, 8, 16, 32]
}
