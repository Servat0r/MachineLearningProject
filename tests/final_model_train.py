# Training (and save) of the final model with results on the blind test set
from __future__ import annotations
from tests.utils import *
from core.utils.types import *
from core.data import *
from core.callbacks import InteractiveLogger, TrainingCSVLogger, TestSetMonitor
from core.metrics import MEE
from time import perf_counter
import json
import platform
from core.model_selection import ParameterList
from datetime import datetime


def model_training_cup(
        dataset_directory_path: str = '../datasets/cup', shuffle_once=True,
        shuffle_seed=0, config_file_path='../results/results_best_2perc_models.json',
        dir_path='../results', training_csv_log_file_name='final_model_train_log.csv',
        blind_test_set_results_name='blind_test_set_predictions.csv',
        final_model_save_path='final_model.model', dtype=np.float32,
):
    # Read cup dataset
    train_data, train_targets, int_test_set_data, int_test_set_targets, cup_test_set_data = read_cup(
        use_internal_test_set=True, directory_path=dataset_directory_path,
        shuffle_once=shuffle_once, shuffle_seed=shuffle_seed, dtype=dtype
    )

    # Load best configuration
    with open(config_file_path, 'r') as fp:
        configurations = json.load(fp)

    # Create model, optimizer, loss and callbacks and compile model
    best_configuration = configurations[0]['config']
    model, optimizer, loss, callbacks = ParameterList().convert(best_configuration)
    model.compile(optimizer, loss, metrics=[MEE()])

    test_set_monitor = TestSetMonitor(
        int_test_set_data, int_test_set_targets, metrics=[MEE()], max_epochs=best_configuration['max_epoch']
    )
    callbacks = [] if callbacks is None else callbacks
    callbacks.append(test_set_monitor)

    logging_callbacks = [
        InteractiveLogger(), TrainingCSVLogger(
            train_directory_path=dir_path, train_file_name=training_csv_log_file_name,
        ),
    ]
    callbacks.extend(logging_callbacks)

    # Create datasets and dataloaders
    train_dataset = ArrayDataset(train_data, train_targets)
    train_dataloader = DataLoader(train_dataset, batch_size=best_configuration['minibatch_size'], shuffle=True)

    crt = perf_counter()
    history = model.train(
        train_dataloader, max_epochs=best_configuration['max_epoch'], callbacks=callbacks,
    )
    crt = perf_counter() - crt
    # Save time and other system info
    with open(os.path.join(dir_path, 'stats.txt'), 'w') as fp:
        uname = platform.uname()
        print(f"Elapsed time = {crt} seconds", file=fp)
        print(f"Available cores: {os.cpu_count()}", file=fp)
        print(f"System: {uname.system}", file=fp)
        print(f"Machine: {uname.machine}", file=fp)
        print(f"Processor: {uname.processor}", file=fp)

    # Plot Loss curve
    plot_data(
        os.path.join(dir_path, 'cup_loss.png'), history['loss'], 'Development Set',
        test_data=test_set_monitor['loss'], test_plot_label='Internal Test Set',
        n_epochs=len(history), ylabel='Loss (MSE)',
    )

    # Plot MEE curve
    plot_data(
        os.path.join(dir_path, 'cup_MEE.png'), history['MEE'], 'Development Set',
        test_data=test_set_monitor['MEE'], test_plot_label='Internal Test Set',
        n_epochs=len(history), ylabel='Mean Euclidean Error (MEE)',
    )

    # Save Test Set log
    ts_loss_data, ts_mee_data = test_set_monitor['loss'], test_set_monitor['MEE']
    with open(os.path.join(dir_path, 'internal_test_set_log.csv'), 'w') as fp:
        print('loss', 'MEE', file=fp, sep=',')
        for i in range(len(test_set_monitor)):
            print(ts_loss_data[i], ts_mee_data[i], sep=',', file=fp)
    """
    plot_metrics(
        history, {
            'loss': 'Development Set',
            'Val_loss': 'Internal Test Set',
        }, os.path.join(dir_path, 'cup_loss.png'),
        len(history), ylabel='Loss (MSE)'
    )
    plot_metrics(
        history, {
            'MEE': 'Development Set',
            'Val_MEE': 'Internal Test Set',
        }, os.path.join(dir_path, 'cup_MEE.png'),
        len(history), ylabel='Mean Euclidean Error (MEE)')
    """

    # Set to test, then save model
    model.set_to_test()
    model.save(
        os.path.join(dir_path, final_model_save_path),
        include_compile_objs=False, include_history=False
    )

    blind_test_set_outputs = model.predict(cup_test_set_data)
    today = datetime.today()
    with open(os.path.join(dir_path, blind_test_set_results_name), 'w') as fp:
        # print("# Salvatore Correnti Gaetano Nicassio Alberto L'Episcopo", file=fp)
        # print("# Cool Generation", file=fp)
        # print("# ML-CUP22", file=fp)
        # print(f"# {today.day}/{today.month}/{today.year}", file=fp)
        for index, outputs in enumerate(blind_test_set_outputs):
            print(index+1, *outputs, sep=',', file=fp)


if __name__ == '__main__':
    model_training_cup()

