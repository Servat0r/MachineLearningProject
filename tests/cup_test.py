# A test with CUP data for checking correctness
from __future__ import annotations
from tests.utils import *
from core.utils.types import *
from core.data import *
from core.callbacks import InteractiveLogger, TrainingCSVLogger, EarlyStopping
from core.metrics import MEE, RMSE, MSE, MAE
import core.utils as cu
import core.modules as cm
from core.model_selection import Holdout


MAX_EPOCHS = 500
SEED = None


def test_cup_once(
        use_internal_test_set=True, directory_path: str = '../datasets/cup',
        internal_test_set_size=0.1, shuffle_once=True, dtype=np.float32, num_iterations=1
):
    # Read cup dataset
    train_data, train_targets, int_test_set_data, int_test_set_targets, cup_test_set_data = read_cup(
        use_internal_test_set, directory_path, internal_test_set_size, shuffle_once, shuffle_seed=0, dtype=dtype
    )

    # Use Holdout split once
    eval_data, eval_targets = None, None
    cross_validator = Holdout()
    for train_values, eval_values in cross_validator.split(train_data, train_targets, shuffle=True, random_state=0,
                                                           validation_split_percentage=0.25):
        train_data, train_targets = train_values
        eval_data, eval_targets = eval_values

    results = []
    for i in range(num_iterations):
        # Create datasets and dataloaders
        train_dataset = ArrayDataset(train_data, train_targets)
        eval_dataset = ArrayDataset(eval_data, eval_targets)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, shuffle_seed=0)
        eval_dataloader = DataLoader(eval_dataset, batch_size=len(eval_dataset))

        # Create model, optimizer, loss etc
        model = cm.Model([
            cm.Input(),
            cm.Dense(
                9, 16, cm.Sigmoid(), weights_initializer=cu.FanInitializer(16, seed=SEED),
                weights_regularizer=cm.L2Regularizer(1e-6), biases_regularizer=cm.L2Regularizer(1e-6),
            ),
            cm.Dense(
                16, 16, cm.Tanh(), weights_initializer=cu.FanInitializer(16, seed=SEED),
                weights_regularizer=cm.L2Regularizer(1e-6), biases_regularizer=cm.L2Regularizer(1e-6),
            ),
            cm.Linear(
                16, 2, weights_initializer=cu.FanInitializer(2, seed=SEED),
                weights_regularizer=cm.L2Regularizer(1e-6), biases_regularizer=cm.L2Regularizer(1e-6),
            ),
        ])
        lr_decay_scheduler = cm.LinearDecayScheduler(1e-2, 1e-4, 500, 8)
        optimizer = cm.SGD(lr=1e-2, lr_decay_scheduler=lr_decay_scheduler, momentum=0.6)
        loss = cm.MSELoss(const=1.0, reduction='mean')

        # Compile and execute
        model.compile(optimizer, loss, metrics=[MEE(), MAE(), RMSE(), MSE(const=1.0)])

        history = model.train(
            train_dataloader, eval_dataloader, max_epochs=MAX_EPOCHS, callbacks=[
                # EarlyStopping(monitor='Val_MEE', mode='min', min_delta=1e-5, patience=200, return_best_result=True),
                InteractiveLogger(), TrainingCSVLogger(train_file_name='cup_train_log.csv')
            ]
        )
        last_mee_value = history['Val_MEE'][len(history)-1]
        print(last_mee_value)
        results.append(last_mee_value)
        plot_metrics(history, ['loss', 'Val_loss'], './loss.png', len(history), ylabel='Loss')
        plot_metrics(history, ['MEE', 'Val_MEE'], './MEE.png', len(history), ylabel='MEE')
        plot_metrics(history, ['MSE', 'Val_MSE'], './MSE.png', len(history), ylabel='MSE')
    return results


if __name__ == '__main__':
    results = test_cup_once(num_iterations=3)
    print(results)
    print(np.mean(results))
