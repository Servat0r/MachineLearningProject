import os
import typer
from tests.monks_tests import *
from tests.cup_search_test import *


app = typer.Typer()


_MONK_PLOT_SAVE_DIR_PATH_HELP = "Relative path for the directory in which to save the resulting plots. " \
                           "Defaults to 'results/monks'"
_MONK_PLOT_LR_HELP = "Learning rate. Defaults to 1e-1 for the first two MONK problems and 1e-2 for the third."
_MONK_PLOT_MOMENTUM_HELP = "Momentum. Defaults to 0.0 (float)."

_CUP_GS_DIR_PATH_HELP = 'Path of the directory in which the file of the grid search is contained.'
_CUP_GS_FILE_PATH_HELP = 'Name of the file for the grid search.'
_CUP_GS_METRIC_HELP = "One of `mee`, `mse`, `rmse`, `mae` as metric for the grid search."
_CUP_GS_CV_HELP = 'One of `holdout`, `kfold` as cross validator for the grid search.'
_CUP_GS_VAL_SPLIT_HELP = 'Validation split percentage (as 0.<perc>) to use in Holdout split.'
_CUP_GS_FOLDS_HELP = 'Number of folds of to use for KFold cross-validation.'
_CUP_GS_SAVE_ALL_HELP = 'Whether or not to save all the configurations of the grid search.'
_CUP_GS_SAVE_BEST_HELP = 'If specified, saves only the given number of models from the best ones.'
_CUP_GS_SAVE_DIR_HELP = 'Relative path of the directory in which to save the results of the search.'
_CUP_GS_NJOBS_HELP = 'Number of worker processes to launch. Defaults to `os.cpu_count()`.'


default_monk_parameters = {
    1: {
        'lr': 0.2,
        'momentum': 0.0,
    },
    2: {
        'lr': 1e-1,
        'momentum': 0.0,
    },
    3: {
        'lr': 1e-2,
        'momentum': 0.0,
    }
}


@app.command(name='monk')
def monk(
        number: int,
        plot_dir_path=typer.Option('results/monks', help=_MONK_PLOT_SAVE_DIR_PATH_HELP),
        lr=typer.Option(None, help=_MONK_PLOT_LR_HELP),
        momentum=typer.Option(None, help=_MONK_PLOT_MOMENTUM_HELP),
):
    """
    Executes the MONK test specified by the `number` parameter.
    :param number: Either 1, 2, or 3, specifying which MONK test to execute.
    :param plot_dir_path: Directory in which to save the resulting plots.
    :param lr: Learning rate (MUST be a float).
    :param momentum: Momentum (MUST be a float).
    """
    if not 1 <= number <= 3:
        raise ValueError(f"Illegal value for 'number': expected one of {{1, 2, 3}}, got {number}")
    else:
        if lr is None:
            lr = default_monk_parameters[number]['lr']
        if momentum is None:
            momentum = default_monk_parameters[number]['momentum']
        print(lr, momentum, type(lr))
        lr, momentum = float(lr), float(momentum)
        plot_save_paths = [
            os.path.join(plot_dir_path, f"monk{number}_losses.png"),
            os.path.join(plot_dir_path, f"monk{number}_accuracy.png"),
        ]
        model_save_path = os.path.join(plot_dir_path, f"monk{number}_model.model")
        if number == 1:
            test_monk1(
                lr=lr, momentum=momentum, plot_save_paths=plot_save_paths,
                dir_path='datasets/monks', model_save_path=model_save_path,
                csv_save_path=plot_dir_path,
            )
        elif number == 2:
            test_monk2(
                lr=lr, momentum=momentum, plot_save_paths=plot_save_paths,
                dir_path='datasets/monks', model_save_path=model_save_path,
                csv_save_path=plot_dir_path,
            )
        else:
            test_monk3(
                lr=lr, momentum=momentum, plot_save_paths=plot_save_paths,
                dir_path='datasets/monks', model_save_path=model_save_path,
                csv_save_path=plot_dir_path,
            )


@app.command(name='cup-grid')
def cmd_cup_grid_search(
        file_paths: list[str],
        dir_path=typer.Option('search', help=_CUP_GS_DIR_PATH_HELP),
        metric=typer.Option('mee', help=_CUP_GS_METRIC_HELP),
        cross_validator=typer.Option('holdout', help=_CUP_GS_CV_HELP),
        val_split=typer.Option(0.25, help=_CUP_GS_VAL_SPLIT_HELP),
        folds=typer.Option(5, help=_CUP_GS_FOLDS_HELP),
        save_all=typer.Option(True, help=_CUP_GS_SAVE_ALL_HELP),
        save_best=typer.Option(0, help=_CUP_GS_SAVE_BEST_HELP),
        save_dir_path=typer.Option('results', help=_CUP_GS_SAVE_DIR_HELP)
):
    """
    Executes the grid search on the parameters specified in the given configuration file
    and saves the results in the folder specified in `save_dir_path`.
    """
    val_split, folds, save_all, save_best = float(val_split), int(folds), bool(save_all), int(save_best)
    metric = __convert_metric(metric)
    cross_validator = __convert_cv(cross_validator, folds)
    print(file_paths)
    for file_path in file_paths:
        if isinstance(cross_validator, cv.Holdout):
            cup_grid_search(
                dir_path, file_path, metric, cross_validator, save_all, save_best, save_dir_path=save_dir_path,
                dataset_dir_path='datasets/cup', validation_split_percentage=val_split
            )
        else:
            cup_grid_search(
                dir_path, file_path, metric, cross_validator, save_all, save_best,
                dataset_dir_path='datasets/cup', save_dir_path=save_dir_path,
            )


@app.command(name='cup-sequential')
def cmd_cup_sequential_search(
        file_paths: list[str],
        dir_path=typer.Option('search', help=_CUP_GS_DIR_PATH_HELP),
        metric=typer.Option('mee', help=_CUP_GS_METRIC_HELP),
        cross_validator=typer.Option('kfold', help=_CUP_GS_CV_HELP),
        val_split=typer.Option(0.25, help=_CUP_GS_VAL_SPLIT_HELP),
        folds=typer.Option(17, help=_CUP_GS_FOLDS_HELP),
        save_all=typer.Option(True, help=_CUP_GS_SAVE_ALL_HELP),
        save_best=typer.Option(0, help=_CUP_GS_SAVE_BEST_HELP),
        save_dir_path=typer.Option('results', help=_CUP_GS_SAVE_DIR_HELP),
        njobs=typer.Option(os.cpu_count(), help=_CUP_GS_NJOBS_HELP),
):
    """
    Executes a sequential search (i.e., on the whole list of given configurations) of the
    configurations specified in the given files and saves the results in the folder specified
    in `save_dir_path`.
    """
    val_split, folds, save_all, save_best = float(val_split), int(folds), bool(save_all), int(save_best)
    njobs = int(njobs)
    njobs = njobs if njobs > 0 else None
    metric = __convert_metric(metric)
    cross_validator = __convert_cv(cross_validator, folds)
    print(file_paths)
    for file_path in file_paths:
        if isinstance(cross_validator, cv.Holdout):
            cup_sequential_search(
                dir_path, file_path, metric, cross_validator, save_all,
                save_best, save_dir_path=save_dir_path, n_jobs=njobs,
                dataset_dir_path='datasets/cup', validation_split_percentage=val_split,
            )
        else:
            cup_sequential_search(
                dir_path, file_path, metric, cross_validator, save_all, save_best,
                dataset_dir_path='datasets/cup', save_dir_path=save_dir_path, n_jobs=njobs,
            )


def __convert_metric(name: str):
    if name == 'mee':
        return MEE()
    elif name == 'mse':
        return MSE()
    elif name == 'rmse':
        return RMSE()
    elif name == 'mae':
        return MAE()
    else:
        raise ValueError(f"Unknown metric '{name}'")


def __convert_cv(name: str, folds: int):
    if name == 'holdout':
        return cv.Holdout()
    elif name == 'kfold':
        return cv.KFold(folds)
    else:
        raise ValueError(f"Unknown cross validator '{name}'")


@app.command()
def run_all():
    print('Command under construction')


if __name__ == '__main__':
    app()
