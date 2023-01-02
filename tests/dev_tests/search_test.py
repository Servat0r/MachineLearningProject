from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from core.utils import *
from core.metrics import *
import core.model_selection.validation as cv
from core.model_selection.search import *


example_param_of_search = {
    'size_hidden_layers': [(4,)],
    'input_dim': [10],
    'output_dim': [2],
    'activation': ['tanh', 'sigmoid'],
    'learning_rate': [1e-3, 1e-4, 1e-5],
    'momentum': [0.0, 0.2, 0.4],
    'regularization': ['none', ('es', 'Val_loss', 1e-3, 50)],  # [('l1', 1e-7), ('l2', 1e-7), ('es', 1e-3)],
    'weights_initialization': [('Uniform', -0.1, 0.1)],
    'max_epoch': [250],
    'minibatch_size': [8, 16, 32]
}

X, y = load_diabetes(return_X_y=True)
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

X_dev = np.expand_dims(X_dev, axis=1)
X_test = np.expand_dims(X_test, axis=1)
y_dev = np.expand_dims(y_dev, axis=(1, 2))
y_test = np.expand_dims(y_test, axis=(1, 2))

# this is for using k-fold
k_fold_grid_search = GridSearch(example_param_of_search, MEE(), cv.KFold(3))
k_fold_grid_search.search(X_dev, y_dev, cv_shuffle=True, cv_random_state=0, epoch_shuffle=True)

# Save the best 4 configs
k_fold_grid_search.save_best(4, file_name='../../core/model_selection/k_fold_best_results.json')

# Load best 4 configs again
# with open('k_fold_best_results.json', 'r') as fp:
#    best_configs = json.load(fp)

best_configs = k_fold_grid_search.results[:4]
best_configs = [config['config'] for config in best_configs]

# this instead is for holdout (in this case on the best models)
# holdout_best_search = FixedCombSearch(best_configs, MEE(), cv.Holdout())
# holdout_best_search.search(
#     X_dev, y_dev, cv_shuffle=True, cv_random_state=0, epoch_shuffle=True, validation_split_percentage=0.2
# )

# Save best model
# holdout_best_search.save_best(1, file_name='final_best_result.json')


# Current number of values is: 3 * 3 * 4 * 4 * 5 * 2 * 2 * 3 * 3 = 25920
param_of_search = {
    "size_hidden_layers": [(8, 8), (8, 4), (4, 4)],  # Queste ce le dividiamo (magari 3 modelli)
    "input_dim": [9],
    "output_dim": [2],
    "activation": ["tanh", "sigmoid", "relu"],  # Queste tre bastano
    "learning_rate": [1e-3, 1e-4, 1e-5, 1e-6],
    "momentum": [0.0, 0.3, 0.6, 0.9],
    "regularization": [
        "none", ("l2", 1e-6), ("l2", 1e-7), ("l2", 1e-8), ("es", "Val_loss", 50)
    ],
    "weights_initialization": [("Uniform", -0.1, 0.1), ("Uniform", -0.5, 0.5)],
    "max_epoch": [500, 1000],
    "minibatch_size": [8, 16, 32],
    # NOTE: For "linear", the first value v is inteded s.t. final learning rate will be comb["learning_rate"] * v
    # e.g. for comb["learning_rate"] = 1e-3, v = 1e-2, there will be end_value = 1e-5 for scheduler
    "decay": ["none", ("linear", 1e-1, 8), ("linear", 1e-2, 8)],
}
