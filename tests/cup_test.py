# A test with CUP data for checking correctness
from __future__ import annotations
from tests.utils import *
from core.utils.types import *
from core.data import *
from core.callbacks import InteractiveLogger, TrainingCSVLogger, EarlyStopping
from core.metrics import MEE, RMSE, MSE
import core.utils as cu
import core.modules as cm
from core.model_selection import Holdout
# from core.transforms import StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def test_cup_once(
        use_internal_test_set=True, directory_path: str = '../datasets/cup',
        internal_test_set_size=0.1, shuffle_once=True, scaler=StandardScaler(), dtype=np.float32,
):
    # Read cup dataset
    train_data, train_targets, int_test_set_data, int_test_set_targets, cup_test_set_data = read_cup(
        use_internal_test_set, directory_path, internal_test_set_size, shuffle_once, dtype
    )

    # Use Holdout split once
    eval_data, eval_targets = None, None
    cross_validator = Holdout()
    for train_values, eval_values in cross_validator.split(train_data, train_targets, shuffle=True, random_state=0,
                                                           validation_split_percentage=0.2):
        train_data, train_targets = train_values
        eval_data, eval_targets = eval_values

    # Create datasets and dataloaders
    train_dataset = ArrayDataset(train_data, train_targets)
    eval_dataset = ArrayDataset(eval_data, eval_targets)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=len(eval_dataset))

    # Create model, optimizer, loss etc
    model = cm.Model([
        cm.Input(),
        cm.Dense(
            9, 16, cm.Tanh(), weights_initializer=cu.RandomUniformInitializer(-0.05, 0.05),
            # biases_initializer=cu.RandomUniformInitializer(-0.01, 0.01),
            # weights_regularizer=cm.L2Regularizer(1e-7), biases_regularizer=cm.L2Regularizer(1e-7),
        ),
        cm.Dense(
            16, 8, cm.Tanh(), weights_initializer=cu.RandomUniformInitializer(-0.05, 0.05),
            # biases_initializer=cu.RandomUniformInitializer(-0.01, 0.01),
            # weights_regularizer=cm.L2Regularizer(1e-7), biases_regularizer=cm.L2Regularizer(1e-7),
        ),
        cm.Linear(
            8, 2, weights_initializer=cu.RandomUniformInitializer(-0.05, 0.05),
            # biases_initializer=cu.RandomUniformInitializer(-0.01, 0.01),
            # weights_regularizer=cm.L2Regularizer(1e-7), biases_regularizer=cm.L2Regularizer(1e-7),
        ),
    ])
    # lr_decay_scheduler=cm.LinearDecayScheduler(1e-3, 1e-4, 1000, 8),
    optimizer = cm.SGD(lr=1e-3, momentum=0.4)
    loss = cm.MSELoss(const=1.0, reduction='mean')

    # Compile and execute
    model.compile(optimizer, loss, metrics=[MEE(), RMSE(), MSE(const=1.0)])

    history = model.train(
        train_dataloader, eval_dataloader, max_epochs=500, callbacks=[
            EarlyStopping(min_delta=1e-4, patience=100, return_best_result=True),
            InteractiveLogger(), TrainingCSVLogger(train_file_name='cup_train_log.csv')
        ]
    )
    last_mse_value = history['Val_MSE'][len(history)-1]
    print(last_mse_value)
    plot_history(0, history, n_epochs=len(history))


if __name__ == '__main__':
    test_cup_once(scaler=MinMaxScaler())
