from core.utils import *
from core.metrics import *
from core.data import *
import core.model_selection.validation as cv
from core.model_selection.search import *


example_param_of_search = {
    'size_hidden_layers': [(16, 8)],  # [(8, 8), (8, 4), (4, 4)],
    'input_dim': [9],
    'output_dim': [2],
    'activation': ['tanh', 'sigmoid', 'relu'],
    'learning_rate': [1e-3, 1e-4, 1e-5],
    'momentum': [0.0, 0.4, 0.8],
    'regularization': [
        'none', ["l2", 1e-6], ["l2", 1e-7], ('es', 'Val_loss', 1e-3, 50)
    ],
    'weights_initialization': [('Uniform', -0.1, 0.1)],
    'max_epoch': [500, 1000],
    'minibatch_size': [8, 16],
    'decay': ['none', ('linear', 1e-1, 8)]
}

# Read cup dataset
train_data, train_targets, int_test_set_data, int_test_set_targets, cup_test_set_data = read_cup(
    use_internal_test_set=True, directory_path='../../datasets/cup', internal_test_set_size=0.2, shuffle_once=True,
)

holdout_grid_search = GridSearch(example_param_of_search, MSE(const=1.0), cv.Holdout())
holdout_grid_search.search(
    train_data, train_targets, cv_shuffle=True, cv_random_state=0, epoch_shuffle=True,
    validation_split_percentage=0.25,
)

holdout_grid_search.save_best(4, file_name='cup_holdout_best_results.json')
print(holdout_grid_search.results[0])
print('Test Finished!')
