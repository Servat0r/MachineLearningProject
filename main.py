from tests import *


def base_tests(*test_nums: int):
    if 0 in test_nums:
        test_separated()


def fc_minibatch_model_tests(*test_nums: int):
    if 0 in test_nums:
        test_fully_connected_minibatch_model(n_epochs=250, mb_size=10, func=arange_square_data)
    if 1 in test_nums:
        test_fully_connected_minibatch_model(n_epochs=250, mb_size=10, func=arange_square_data, epoch_shuffle=False)

    if 2 in test_nums:
        test_fully_connected_minibatch_model(n_epochs=250, mb_size=10, func=randn_sqrt_data)
    if 3 in test_nums:
        test_fully_connected_minibatch_model(n_epochs=250, mb_size=10, func=randn_sqrt_data, epoch_shuffle=False)


def fc_minibatch_model_regularization(*test_nums: int):
    if 0 in test_nums:
        test_fully_connected_minibatch_model_with_regularizations(
            n_epochs=20, mb_size=10, func=arange_square_data, l1_regularizer=1e-6, l2_regularizer=1e-7,
            start_plot_epoch=0, lr=1e-4,
        )
    if 1 in test_nums:
        test_fully_connected_minibatch_model_with_regularizations(
            n_epochs=20, mb_size=10, func=arange_square_data, l1_regularizer=1e-6, l2_regularizer=1e-7,
            start_plot_epoch=0, lr=1e-4, epoch_shuffle=False,
        )

    if 2 in test_nums:
        test_fully_connected_minibatch_model_with_regularizations(
            n_epochs=20, mb_size=10, func=randn_sqrt_data, l1_regularizer=1e-5, l2_regularizer=1e-6,
        )
    if 3 in test_nums:
        test_fully_connected_minibatch_model_with_regularizations(
            n_epochs=20, mb_size=10, func=randn_sqrt_data, epoch_shuffle=False,
            l1_regularizer=1e-5, l2_regularizer=1e-6,
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
            n_epochs=20, mb_size=10, func=arange_square_data, lr=1e-4, momentum=0.9,
            # max_iter = n_epochs * mb_size
            lr_scheduler=cm.LinearDecayScheduler(
                start_value=1e-4, end_value=1e-5, max_iter=20, round_val=6,
            ),
            # lr_scheduler=cm.IterBasedDecayScheduler(start_value=0.01, decay=0.001),
            # lr_scheduler=cm.ExponentialDecayScheduler(start_value=1e-3, alpha=1e-3),
            l1_regularizer=1e-6, l2_regularizer=1e-7,
            # arange_sine_data extra args for validation set
            start=N_SAMPLES,
        )
    if 3 in test_nums:
        test_fc_minibatch_model_with_regularizations_lrscheduler(
            n_epochs=20, mb_size=10, func=arange_square_data, lr=1e-4, momentum=0.9, epoch_shuffle=False,
            # max_iter = n_epochs * mb_size
            lr_scheduler=cm.LinearDecayScheduler(
                start_value=1e-4, end_value=1e-5, max_iter=20, round_val=6,
            ),
            # lr_scheduler=cm.IterBasedDecayScheduler(start_value=0.01, decay=0.001),
            # lr_scheduler=cm.ExponentialDecayScheduler(start_value=1e-3, alpha=1e-3),
            l1_regularizer=1e-6, l2_regularizer=1e-7,
            # arange_sine_data extra args for validation set
            start=N_SAMPLES,
        )


if __name__ == '__main__':
    base_tests(-1)
    fc_minibatch_model_tests(-1)
    fc_minibatch_model_regularization(0, 1)
    fc_minibatch_model_regularization_lrdecay(2, 3)
    exit(0)
