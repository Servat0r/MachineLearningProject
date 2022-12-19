import os
import sys
import json
from core.utils import *
from core.metrics import *
from core.data import *
import core.model_selection.validation as cv
from core.model_selection.search import *


def cup_grid_search(
        config_directory_name: str, config_file_name: str, metric: Metric = MEE(),
        cross_validator: cv.Validator = cv.Holdout(), save_all: bool = True,
        save_best: int = None, *args, **kwargs
):
    config_file_path = os.path.join(config_directory_name, config_file_name)
    with open(config_file_path, 'r') as fp:
        params_of_search = json.load(fp)

    total_number_of_configurations = np.prod([len(val) for val in params_of_search.values()]).item()
    print(f"Grid Search with {total_number_of_configurations} total number of configurations")

    # Read cup dataset
    train_data, train_targets, int_test_set_data, int_test_set_targets, cup_test_set_data = read_cup(
        use_internal_test_set=True, directory_path='../../datasets/cup', internal_test_set_size=0.1, shuffle_once=True,
    )

    grid_search = GridSearch(params_of_search, metric, cross_validator)
    grid_search.search(
        train_data, train_targets, cv_shuffle=True, cv_random_state=0, epoch_shuffle=True, *args, **kwargs
    )

    if save_all:
        grid_search.save_all(directory_path='../../results', file_name=f"results_MSE_{config_file_name}")
    else:
        grid_search.save_best(save_best, directory_path='../../results', file_name=f"results_MSE_{config_file_name}")

    print('Test Finished!')


if __name__ == '__main__':
    name = sys.argv[1]
    if name == 'salvatore':
        cup_grid_search('../../search', 'coarse_gs_1_salvatore.json', save_all=True, validation_split_percentage=0.25)
    elif name == 'gaetano':
        # cup_grid_search('../../search', 'coarse_gs_1_gaetano.json', save_all=True, validation_split_percentage=0.25)
        # cup_grid_search('../../search', 'coarse_gs_2_gaetano.json', save_all=True, validation_split_percentage=0.25)
        # cup_grid_search('../../search', 'coarse_gs_3_gaetano.json', save_all=True, validation_split_percentage=0.25)
        cup_grid_search('../../search', 'coarse_gs_4_gaetano.json', save_all=True, validation_split_percentage=0.25)
    elif name == 'alberto':
        pass
