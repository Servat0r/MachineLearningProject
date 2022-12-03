
### Description
Final project for the Machine Learning course at University of Pisa, a.y. 2022/23.
The project consists in implementing a framework for building Neural Networks from scratch, and in particular
"classic" MLP with backpropagation and stochastic gradient descent training.
Framework is tested with the MONK datasets () for classification and the "ML-CUP" dataset,
an internal competition between the students of the current a.y., for regression.

### Directory Structure
```
Machine Learning Project
|___ core
|    |___ callbacks
|    |___ data
|    |___ metrics
|    |___ modules
|    |___ utils
|    |___ functions.py
|    |___ transforms.py
|___ tests
|    |___ dev_tests
|    |___ keras_tests
|    |___ monks_tests.py
|    |___ utils.py
|___ datasets
|    |___ monks
|    |___ cup
|___ results
     |___ monks
     |___ cup
```

### Content Description
`core.modules.layers`: implementation of layers (Input, Linear, Activation, Dense);

`core.modules.losses`: implementation of losses (MSE, CrossEntropy);

`core.modules.optimizers`: implementation of optimizers (SGD with momentum);

`core.modules.regularization`: implementation of regularizers (L1, L2, L1L2);

`core.modules.schedulers`: functions (Schedulers) for learning rate decay;

`core.modules.model`: implementation of Model object for encapsulating layers, losses,
optimizers etc.

`core.data`: implementation of `Datasets` and `DataLoaders` (inspired by PyTorch) for retrieving
and loading data; file `core.data.commons.py` contains loading utilities for MONK and CUP datasets;

`core.metrics`: implementation of metrics for monitoring progress during training. Available metrics
are: `MeanSquaredError`, `MeanEuclideanError` and `RootMeanSquaredError` for regression;
`Accuracy`, `BinaryAccuracy`, `CategoricalAccuracy`, `SparseCategoricalAccuracy` for classification;
`Timing` for monitoring time;

`core.callbacks`: implementation of callbacks to be inserted in the training / evaluation / test loop.
Some available ones are: `EarlyStopping`, `TrainingCSVLogger`, `ModelCheckpoint`.

`core.functions`: common functions (SquaredError, CrossEntropy, Softmax) for usage in layers/losses/metrics.

`core.transforms`: some transformations for the data, e.g. `OneHotEncoder`.

Framework interface is inspired mostly by Keras, and somewhat by PyTorch and scikit-learn: in particular,
the Layer and Model interface partially mimic the Keras correspondent. For example:

`Model Creation`:
```
import core.modules as cm

model = core.modules.Model([
    cm.Input(),
    cm.Dense(17, 4, cm.Tanh()),
    cm.Linear(4, 1)
])
```
The above creates an MLP made up by an input layer, a hidden layer with 4 units
that accepts inputs of size 17, and an output layer with 1 unit and no (= identity)
activation, with an intermediate activation layer with `tanh`.

`Model Setup and Training`:
```
from core.metrics import MEE, RMSE
from core.callbacks import EarlyStopping

# Suppose train_dataloader, eval_dataloader are training
# and evaluation DataLoaders.

optimizer = cm.SGD(lr=1e-3)
loss = cm.MSELoss(const=1.)
model.compile(optimizer, loss, metrics=[MEE(), RMSE()]
history = model.train(
    train_dataloader, eval_dataloader, n_epochs=100,
    callbacks=[EarlyStopping('Val_MEE', patience=2)]
)
```
The above creates a SGD optimizer and a MSE loss, configures the model
for training with MeanEuclideanError and RootMeanSquaredError as metrics
for both training and validation sets, and trains the model on the given
data for at most 100 epochs and with EarlyStopping strategy monitoring
the MEE on validation set.