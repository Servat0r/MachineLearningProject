# Tests for saving and loading a model
from core.data import DataLoader
from tests.utils import *
import core.utils as cu
import core.modules as cm


def check_same_params(model1: cm.Model, model2: cm.Model):
    try:
        for layer1, layer2 in zip(model1.layers, model2.layers):
            if isinstance(layer1, cm.Dense):
                assert isinstance(layer2, cm.Dense)
                # First check that shape is equal
                assert layer1.linear.weights.shape == layer2.linear.weights.shape
                assert layer1.linear.biases.shape == layer2.linear.biases.shape
                # Then check that values are equal
                assert np.allclose(layer1.linear.weights, layer2.linear.weights)
                assert np.allclose(layer1.linear.biases, layer2.linear.biases)
            elif isinstance(layer1, cm.Linear):
                assert isinstance(layer2, cm.Linear)
                # First check that shape is equal
                assert layer1.weights.shape == layer2.weights.shape
                assert layer1.biases.shape == layer2.biases.shape
                # Then check that values are equal
                assert np.allclose(layer1.weights, layer2.weights)
                assert np.allclose(layer1.biases, layer2.biases)
        return True
    except AssertionError as ae:
        print(ae)
        return False


def __get_model(in_dim=16, hidden_dim=8, out_dim=1, low=-0.1, high=0.1, l1_lambda=1e-4, l2_lambda=1e-4):
    w_init = cu.RandomUniformInitializer(low, high)
    return cm.Model([
        cm.Dense(
            in_dim, hidden_dim, cm.Tanh(), weights_initializer=w_init,
            weights_regularizer=cm.L1Regularizer(l1_lambda=l1_lambda)
        ),
        cm.Dense(
            hidden_dim, hidden_dim, cm.Tanh(), weights_initializer=w_init,
            weights_regularizer=cm.L2Regularizer(l2_lambda=l2_lambda)
        ),
        cm.Linear(
            hidden_dim, out_dim, weights_initializer=w_init,
            weights_regularizer=cm.L1L2Regularizer()
        ),
    ])


def test_save_load_parameters(fpath='parameters.params'):
    model1 = __get_model(in_dim=16, hidden_dim=8, out_dim=1, low=-0.5, high=0.5, l1_lambda=1e-4, l2_lambda=1e-4)
    model1.save_parameters(fpath)
    model2 = __get_model(in_dim=32, hidden_dim=32, out_dim=16, low=-3.0, high=3.0, l1_lambda=1., l2_lambda=1.)
    assert not check_same_params(model1, model2)
    model2.load_parameters(fpath)
    assert check_same_params(model1, model2)
    print(f'{test_save_load_parameters} passed')


def test_save_load_untrained_model(fpath='untrained_model.model'):
    model1 = __get_model(16, 8, 1, low=-0.5, high=0.5, l1_lambda=1e-4, l2_lambda=1e-3)
    model1.save(fpath)
    model2 = cm.Model.load(fpath)
    assert check_same_params(model1, model2)
    print(f'{test_save_load_untrained_model} passed')


def test_save_load_trained_model(fpath='trained_model.model'):
    model1 = __get_model(INPUT_DIM, HIDDEN_SIZE, OUTPUT_DIM)
    x, y, train_dataset, _ = generate_dataset(func=arange_square_data)
    train_dataloader = DataLoader(train_dataset)

    loss = cm.MSELoss()
    optimizer = cm.SGD(lr=1e-3, momentum=0.9)
    model1.compile(optimizer=optimizer, loss=loss)

    model1.train(train_dataloader, n_epochs=10)
    model1.save(fpath)
    model2 = cm.Model.load(fpath)
    assert check_same_params(model1, model2)
    print(f'{test_save_load_trained_model} passed')


def test_save_load_predict_trained_model(fpath='trained_model.model'):
    model1 = __get_model(INPUT_DIM, HIDDEN_SIZE, OUTPUT_DIM)
    x, y, train_dataset, _ = generate_dataset(func=arange_square_data, samples=N_SAMPLES)
    train_dataloader = DataLoader(train_dataset)

    loss = cm.MSELoss()
    optimizer = cm.SGD(lr=1e-3, momentum=0.9)
    model1.compile(optimizer=optimizer, loss=loss)

    model1.train(train_dataloader, n_epochs=10)
    model1.save(fpath)
    model2 = cm.Model.load(fpath)
    x_test, y_eval = arange_square_data(
        samples=N_SAMPLES//5, start=EVAL_START, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM
    )
    y_hat_1 = model1.predict(x_test)
    y_hat_2 = model2.predict(x_test)
    assert np.allclose(y_hat_1, y_hat_2)
    loss_val_1, loss_val_2 = loss(y_hat_1, y_eval), loss(y_hat_2, y_eval)
    assert np.allclose(loss_val_1, loss_val_2)
    print(f'{test_save_load_predict_trained_model} passed')


# todo we should also add a test for models DURING training (this requires to modify pickle serialization!)


if __name__ == '__main__':
    test_save_load_parameters()
    test_save_load_untrained_model()
    test_save_load_trained_model()
    test_save_load_predict_trained_model()
