import json
from typing import Iterable


def rank_hyperparameter(config_file_path: str, hyperparameter_name: str):
    with open(config_file_path, 'r') as fp:
        raw_data = json.load(fp)
    configs_data = [item['config'] for item in raw_data]
    results = {}
    length = len(configs_data)
    for i in range(length):
        config = configs_data[i]
        value = config.get(hyperparameter_name, None)
        if isinstance(value, Iterable):  # list is not hashable
            value = tuple(value)
        score = length - i
        if value not in results:
            results[value] = score
        else:
            results[value] += score
    return results


__all__ = ['rank_hyperparameter']
