from tests import *
import core.metrics as cmt


def base_tests(*test_nums: int):
    if 0 in test_nums:
        test_separated()


def fc_minibatch_model_tests(*test_nums: int):
    if 0 in test_nums:
        test_fully_connected_minibatch_model(
            n_epochs=250, mb_size=10, func=arange_square_data, epoch_shuffle=True, func_args={'start': EVAL_START},
        )
    if 1 in test_nums:
        test_fully_connected_minibatch_model(
            n_epochs=250, mb_size=10, func=arange_square_data, epoch_shuffle=False, func_args={'start': EVAL_START},
        )

    if 2 in test_nums:
        test_fully_connected_minibatch_model(
            n_epochs=250, mb_size=10, func=randn_sqrt_data, epoch_shuffle=True, func_args={},
        )
    if 3 in test_nums:
        test_fully_connected_minibatch_model(
            n_epochs=250, mb_size=10, func=randn_sqrt_data, epoch_shuffle=False, func_args={},
        )


def fc_minibatch_model_regularization(*test_nums: int):
    if 0 in test_nums:
        test_fully_connected_minibatch_model_with_regularizations(
            n_epochs=250, mb_size=10, func=arange_square_data, l1_regularizer=1e-6, l2_regularizer=1e-7,
            start_plot_epoch=0, lr=1e-4, epoch_shuffle=True, func_args={'start': EVAL_START},
        )
    if 1 in test_nums:
        test_fully_connected_minibatch_model_with_regularizations(
            n_epochs=250, mb_size=10, func=arange_square_data, l1_regularizer=1e-6, l2_regularizer=1e-7,
            start_plot_epoch=0, lr=1e-4, epoch_shuffle=False, func_args={'start': EVAL_START},
        )

    if 2 in test_nums:
        test_fully_connected_minibatch_model_with_regularizations(
            n_epochs=20, mb_size=10, func=randn_sqrt_data, l1_regularizer=1e-5,
            l2_regularizer=1e-6, func_args={},
        )
    if 3 in test_nums:
        test_fully_connected_minibatch_model_with_regularizations(
            n_epochs=20, mb_size=10, func=randn_sqrt_data, epoch_shuffle=False,
            l1_regularizer=1e-5, l2_regularizer=1e-6, func_args={},
        )


def fc_minibatch_model_regularization_lrdecay(*test_nums: int):
    if 0 in test_nums:
        test_fc_minibatch_model_with_regularizations_lrscheduler(
            n_epochs=20, mb_size=10, func=randn_sqrt_data, lr=1e-3, momentum=0.,
            # max_iter = n_epochs * mb_size
            # lr_scheduler=cm.LinearDecayScheduler(start_value=0.01, end_value=0.005, max_iter=100*(N_SAMPLES//50)),
            # lr_scheduler=cm.IterBasedDecayScheduler(start_value=0.01, decay=0.001),
            lr_scheduler=cm.ExponentialDecayScheduler(start_value=1e-3, alpha=1e-3),
            l1_regularizer=1e-5, l2_regularizer=1e-6,
        )
    if 1 in test_nums:
        test_fc_minibatch_model_with_regularizations_lrscheduler(
            n_epochs=20, mb_size=10, func=randn_sqrt_data, lr=1e-3, momentum=0., epoch_shuffle=False,
            # max_iter = n_epochs * mb_size
            # lr_scheduler=cm.LinearDecayScheduler(start_value=0.01, end_value=0.005, max_iter=100*(N_SAMPLES//50)),
            # lr_scheduler=cm.IterBasedDecayScheduler(start_value=0.01, decay=0.001),
            lr_scheduler=cm.ExponentialDecayScheduler(start_value=1e-3, alpha=1e-3),
            l1_regularizer=1e-5, l2_regularizer=1e-6,
        )

    if 2 in test_nums:
        test_fc_minibatch_model_with_regularizations_lrscheduler(
            n_epochs=100, mb_size=10, func=arange_square_data, lr=1e-4, momentum=0.9,
            # max_iter = n_epochs * mb_size
            lr_scheduler=cm.LinearDecayScheduler(
                start_value=1e-4, end_value=1e-5, max_iter=100, round_val=6,
            ),
            # lr_scheduler=cm.IterBasedDecayScheduler(start_value=0.01, decay=0.001),
            # lr_scheduler=cm.ExponentialDecayScheduler(start_value=1e-3, alpha=1e-3),
            l1_regularizer=1e-6, l2_regularizer=1e-7,
            # arange_sine_data extra args for validation set
            func_args={'start': EVAL_START},
        )
    if 3 in test_nums:
        test_fc_minibatch_model_with_regularizations_lrscheduler(
            n_epochs=150, mb_size=10, func=arange_square_data, lr=1e-4, momentum=0.9, epoch_shuffle=False,
            # max_iter = n_epochs * mb_size
            lr_scheduler=cm.LinearDecayScheduler(
                start_value=1e-4, end_value=1e-5, max_iter=150, round_val=6,
            ),
            # lr_scheduler=cm.IterBasedDecayScheduler(start_value=0.01, decay=0.001),
            # lr_scheduler=cm.ExponentialDecayScheduler(start_value=1e-3, alpha=1e-3),
            l1_regularizer=1e-6, l2_regularizer=1e-7,
            # arange_sine_data extra args for validation set
            func_args={'start': EVAL_START},
        )


def fc_minibatch_model_regularization_metrics(*test_nums: int):
    if 0 in test_nums:
        test_fully_connected_minibatch_regularization_metrics(
            n_epochs=250, mb_size=10, func=arange_square_data, l1_regularizer=1e-6, l2_regularizer=1e-7,
            start_plot_epoch=0, lr=1e-4, epoch_shuffle=True, func_args={'start': EVAL_START},
            metrics=[cmt.MEE(), cmt.MSE(), cmt.RMSE()],
        )
    if 1 in test_nums:
        test_fully_connected_minibatch_regularization_metrics(
            n_epochs=50, mb_size=10, func=arange_square_data, l1_regularizer=1e-6, l2_regularizer=1e-7,
            start_plot_epoch=0, lr=1e-4, epoch_shuffle=False, func_args={'start': EVAL_START},
            metrics=[cmt.MEE(), cmt.MSE(), cmt.RMSE()],
        )

    if 2 in test_nums:
        test_fully_connected_minibatch_regularization_metrics(
            n_epochs=20, mb_size=10, func=randn_sqrt_data, l1_regularizer=1e-5,
            l2_regularizer=1e-6, func_args={},
            metrics=[cmt.MEE(), cmt.MSE(), cmt.RMSE()],
        )
    if 3 in test_nums:
        test_fully_connected_minibatch_regularization_metrics(
            n_epochs=20, mb_size=10, func=randn_sqrt_data, epoch_shuffle=False,
            l1_regularizer=1e-5, l2_regularizer=1e-6, func_args={},
            metrics=[cmt.MEE(), cmt.MSE(), cmt.RMSE()],
        )


def fc_minibatch_model_regularization_metrics_logging(*test_nums: int):
    if 0 in test_nums:
        test_fully_connected_regularization_metrics_logging(
            n_epochs=250, mb_size=10, func=arange_square_data, l1_regularizer=1e-6, l2_regularizer=1e-7,
            start_plot_epoch=0, lr=1e-4, epoch_shuffle=True, func_args={'start': EVAL_START},
            metrics=[cmt.MEE(), cmt.RMSE()], train_log_file='train_log.csv', round_val=4,
            # include_mb=True,
        )
    if 1 in test_nums:
        test_fully_connected_regularization_metrics_logging(
            n_epochs=50, mb_size=10, func=arange_square_data, l1_regularizer=1e-6, l2_regularizer=1e-7,
            start_plot_epoch=0, lr=1e-4, epoch_shuffle=False, func_args={'start': EVAL_START},
            metrics=[cmt.MEE(), cmt.MSE(), cmt.RMSE(), cmt.Timing()], train_log_file='train_log.csv',
            round_val=8, # include_mb=True,
        )

    if 2 in test_nums:
        test_fully_connected_regularization_metrics_logging(
            n_epochs=20, mb_size=10, func=randn_sqrt_data, l1_regularizer=1e-5,
            l2_regularizer=1e-6, func_args={}, train_log_file='train_log.csv',
            metrics=[cmt.MEE(), cmt.MSE(), cmt.RMSE(), cmt.Timing()],
            round_val=8, # include_mb=True,
        )
    if 3 in test_nums:
        test_fully_connected_regularization_metrics_logging(
            n_epochs=20, mb_size=10, func=randn_sqrt_data, epoch_shuffle=False,
            l1_regularizer=1e-5, l2_regularizer=1e-6, func_args={},
            metrics=[cmt.MEE(), cmt.MSE(), cmt.RMSE(), cmt.Timing()], train_log_file='train_log.csv',
            round_val=8, # include_mb=True,
        )


if __name__ == '__main__':
    # sbase_tests(0)
    # fc_minibatch_model_tests(0, 1, 2, 3)
    # fc_minibatch_model_regularization(0, 1, 2, 3)
    # fc_minibatch_model_regularization_lrdecay(0, 1, 2, 3)
    # fc_minibatch_model_regularization_metrics(0, 1, 2, 3)
    fc_minibatch_model_regularization_metrics_logging(0)  #, 1, 2, 3)
    exit(0)
