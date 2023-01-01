import typer
from tests.monks_tests import *
from tests.cup_search_test import *
from tests.final_model_train import *


app = typer.Typer()


_MONK_PLOT_SAVE_DIR_PATH_HELP = "Relative path for the directory in which to save the resulting plots. " \
                           "Defaults to 'results/monks'"
_MONK_PLOT_LR_HELP = "Learning rate. Defaults to 0.5 for the first MONK problem, 0.1 for the second and " \
                     "1e-2 for the third."
_MONK_PLOT_MOMENTUM_HELP = "Momentum (float)."
_MONK_PLOT_L2_LAMBDA_HELP = "L2-regularization lambda (float). Used only for MONK-3."

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

_CUP_FINAL_DATASET_PATH = 'Path of the directory in which the CUP dataset is contained.'
_CUP_FINAL_CONFIG_FILE_PATH = 'Relative path of the json file that contains the final model configuration.'
_CUP_FINAL_RESULTS_DIR_PATH = 'Relative path of the folder in which to store the results (model, plots, logs).'
_CUP_FINAL_TRAIN_CSV_FNAME = 'Name of the CSV file for training data logging.'
_CUP_FINAL_BLIND_TS_FNAME = 'Name of the CSV file that will contain the blind test set predictions.'
_CUP_FINAL_MODEL_SAVE_PATH = 'Name of the .model file that will contain the serialized final model.'


default_monk_parameters = {
    1: {
        'lr': 0.5,
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
        num_iterations: int = typer.Option(5, help='Number of trainings to execute'),
        plot_dir_path=typer.Option('results/monks', help=_MONK_PLOT_SAVE_DIR_PATH_HELP),
        lr=typer.Option(None, help=_MONK_PLOT_LR_HELP),
        momentum=typer.Option(None, help=_MONK_PLOT_MOMENTUM_HELP),
        l2_lambda: float = typer.Option(1e-5, help=_MONK_PLOT_L2_LAMBDA_HELP),
):
    """
    Executes the MONK test specified by the `number` parameter.
    :param number: Either 1, 2, or 3, specifying which MONK test to execute.
    :param num_iterations: Number of training cycles to execute.
    :param plot_dir_path: Directory in which to save the resulting plots.
    :param lr: Learning rate (MUST be a float).
    :param momentum: Momentum (MUST be a float).
    :param l2_lambda: L2-regularization lambda value (MUST be a float).
    Used ONLY for MONK-3. Defaults to 0.0.
    """
    if not 1 <= number <= 3:
        raise ValueError(f"Illegal value for 'number': expected one of {{1, 2, 3}}, got {number}")
    else:
        if lr is None:
            lr = default_monk_parameters[number]['lr']
        if momentum is None:
            momentum = default_monk_parameters[number]['momentum']
        print(lr, momentum, type(lr))
        lr, momentum, l2_lambda = float(lr), float(momentum), float(l2_lambda)
        num_iterations = int(num_iterations)
        plot_save_paths = [
            os.path.join(plot_dir_path, f"monk{number}_losses.png"),
            os.path.join(plot_dir_path, f"monk{number}_accuracy.png"),
        ]
        model_save_path = os.path.join(plot_dir_path, f"monk{number}_model.model")
        if number == 1:
            test_monk1(
                lr=lr, momentum=momentum, plot_save_paths=plot_save_paths,
                dir_path='datasets/monks', model_save_path=model_save_path,
                csv_save_path=plot_dir_path, batch_size=2, num_iterations=num_iterations,
            )
        elif number == 2:
            test_monk2(
                lr=lr, momentum=momentum, plot_save_paths=plot_save_paths,
                dir_path='datasets/monks', model_save_path=model_save_path,
                csv_save_path=plot_dir_path, batch_size=2, num_iterations=num_iterations,
            )
        else:
            metrics_to_plot = [
                # Here we are using MSE metric as it corresponds to data loss (i.e., without the penalty term)
                {
                    'MSE': "Training",
                    'Val_MSE': "Test",
                },
                {
                    'BinaryAccuracy': "Training",
                    'Val_BinaryAccuracy': "Test",
                }
            ]
            test_monk3(
                    lr=lr, momentum=momentum, plot_save_paths=plot_save_paths,
                    dir_path='datasets/monks', model_save_path=model_save_path,
                    csv_save_path=plot_dir_path, l2_lambda=l2_lambda, batch_size=2,
                    num_iterations=num_iterations, metrics_to_plot=metrics_to_plot,
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
        save_dir_path=typer.Option('results', help=_CUP_GS_SAVE_DIR_HELP),
        use_test_data: bool = typer.Option(
            False, help='If True, at the end of train cycle records values on internal test set'
        )
):
    """
    Executes the grid search on the parameters specified in the given configuration file
    and saves the results in the folder specified in `save_dir_path`.
    """
    val_split, folds, save_all, save_best = float(val_split), int(folds), bool(save_all), int(save_best)
    use_test_data = bool(use_test_data)
    metric = __convert_metric(metric)
    cross_validator = __convert_cv(cross_validator, folds)
    print(file_paths)
    for file_path in file_paths:
        if isinstance(cross_validator, cv.Holdout):
            cup_grid_search(
                dir_path, file_path, metric, cross_validator, save_all, save_best, save_dir_path=save_dir_path,
                dataset_dir_path='datasets/cup', use_int_test_set=use_test_data, validation_split_percentage=val_split
            )
        else:
            cup_grid_search(
                dir_path, file_path, metric, cross_validator, save_all, save_best,
                dataset_dir_path='datasets/cup', save_dir_path=save_dir_path,
                use_int_test_set=use_test_data
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


@app.command(name='final-train')
def final_train(
        dataset_dir_path: str = typer.Option('datasets/cup', help=_CUP_FINAL_DATASET_PATH),
        config_file_path: str = typer.Option(
            'results/results_best_2perc_models.json', help=_CUP_FINAL_CONFIG_FILE_PATH
        ),
        results_dir_path: str = typer.Option('results', help=_CUP_FINAL_RESULTS_DIR_PATH),
        training_csv_log_file_name: str = typer.Option('final_model_train_log.csv', help=_CUP_FINAL_TRAIN_CSV_FNAME),
        blind_test_set_results_name: str = typer.Option('blind_test_set_predictions.csv', help=_CUP_FINAL_BLIND_TS_FNAME),
        final_model_save_path: str = typer.Option('final_model.model', help=_CUP_FINAL_MODEL_SAVE_PATH),
):
    """
    Executes training of the final selected model by loading its configuration from the results
    file of the cross-validation on the best models. After training, both the plots of the loss
    and the MEE on the development and internal test sets and the final model (as pickle file)
    are saved.
    """
    model_training_cup(
        dataset_dir_path, config_file_path=config_file_path, dir_path=results_dir_path,
        training_csv_log_file_name=training_csv_log_file_name,
        blind_test_set_results_name=blind_test_set_results_name,
        final_model_save_path=final_model_save_path
    )


if __name__ == '__main__':
    app()
