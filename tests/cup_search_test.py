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
        save_best: int = None, dataset_dir_path='../datasets/cup',
        save_dir_path: str = '../results', *args, **kwargs
):
    config_file_path = os.path.join(config_directory_name, config_file_name)
    with open(config_file_path, 'r') as fp:
        params_of_search = json.load(fp)

    total_number_of_configurations = np.prod([len(val) for val in params_of_search.values()]).item()
    print(f"Grid Search with {total_number_of_configurations} total number of configurations")

    # Read cup dataset
    train_data, train_targets, int_test_set_data, int_test_set_targets, cup_test_set_data = read_cup(
        use_internal_test_set=True, directory_path=dataset_dir_path, internal_test_set_size=0.1,
        shuffle_once=True, shuffle_seed=0,
    )

    grid_search = GridSearch(params_of_search, metric, cross_validator)
    search_stats_file = os.path.join(save_dir_path, f"grid_search_stats_{config_file_name.split('.json')[0]}.txt")
    grid_search.search(
        train_data, train_targets, cv_shuffle=True, cv_random_state=0, epoch_shuffle=True,
        search_stats_file=search_stats_file, *args, **kwargs
    )

    if save_all:
        grid_search.save_all(directory_path=save_dir_path, file_name=f"results_{config_file_name}")
    else:
        grid_search.save_best(save_best, directory_path=save_dir_path, file_name=f"results_{config_file_name}")

    print('Test Finished!')


def get_best_models(
        configs_directory_name: str, config_file_names: list[str],
        models_percentage: float, out_file_name: str, key=lambda x: x['mean'],
):
    """
    Retrieves best models from a set of JSON files that contain results of a grid search.
    """
    configurations = []
    for config_file_name in config_file_names:
        config_file_path = os.path.join(configs_directory_name, config_file_name)
        with open(config_file_path, 'r') as fp:
            configurations.extend(json.load(fp))
    models_num = int(models_percentage * len(configurations))
    best_configurations = sorted(configurations, key=key)[0:models_num]
    with open(out_file_name, 'w') as fp:
        json.dump(best_configurations, fp, indent=2)
    print(f'Retrieved and saved {models_num} configurations')


def cup_sequential_search(
        config_directory_name: str, config_file_name: str, metric: Metric = MEE(),
        cross_validator: cv.Validator = cv.KFold(number_of_folds=17), save_all: bool = True,
        save_best: int = None, save_dir_path='../results', dataset_dir_path='../datasets/cup',
        n_jobs=os.cpu_count(), *args, **kwargs
):
    config_file_path = os.path.join(config_directory_name, config_file_name)
    with open(config_file_path, 'r') as fp:
        params_of_search = json.load(fp)

    params_of_search = [data.get('config') for data in params_of_search]

    total_number_of_configurations = len(params_of_search)
    print(f"Sequential Search with {total_number_of_configurations} total number of configurations")

    # Read cup dataset
    train_data, train_targets, int_test_set_data, int_test_set_targets, cup_test_set_data = read_cup(
        use_internal_test_set=True, directory_path=dataset_dir_path, internal_test_set_size=0.1, shuffle_once=True,
    )

    sequential_search = FixedCombSearch(params_of_search, metric, cross_validator)
    search_stats_file = os.path.join(save_dir_path, f"sequential_search_stats_{config_file_name.split('.json')[0]}.txt")
    sequential_search.search(
        train_data, train_targets, cv_shuffle=True, cv_random_state=0, epoch_shuffle=True,
        n_jobs=n_jobs, search_stats_file=search_stats_file, *args, **kwargs
    )

    if save_all:
        sequential_search.save_all(directory_path=save_dir_path, file_name=f"results_{config_file_name}")
    else:
        sequential_search.save_best(save_best, directory_path=save_dir_path, file_name=f"results_{config_file_name}")

    print('Test Finished!')

