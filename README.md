
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

`Train, Validation and Test Dataset setup`
```
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Load and split dataset
X, y = load_diabetes(return_X_y=True)

# Split in development and test set
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Split dev set in training and validation ones
X_train, X_eval, y_train, y_eval = train_test_split(X_dev, y_dev, test_size=0.3, random_state=42, shuffle=True)

# Standardize data format in arrays of shape (l, 1, n)
# l = number of examples; n = input dimension
X_train = np.expand_dims(X_train, axis=1)
X_eval = np.expand_dims(X_eval, axis=1)

y_train = np.expand_dims(y_train, axis=1)
y_eval = np.expand_dims(y_eval, axis=1)

# Now create datasets and dataloaders
train_dataset, eval_dataset = ArrayDataset(X_train, y_train), ArrayDataset(X_eval, y_eval)
train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=epoch_shuffle)
eval_dataloader = DataLoader(eval_dataset, batch_size=len(eval_dataset))
```
The above loads the "breast cancer" dataset using scikit-learn built-in function
`load_breast_cancer`, then split the dataset into train, validation and test ones
(hold-out technique) and creates ArrayDatasets from training and validation data
and their corresponding DataLoaders.

`Model Setup and Training`:
```
from core.metrics import MEE, RMSE
from core.callbacks import EarlyStopping

# Create optimizer and loss
optimizer = cm.SGD(lr=1e-3)
loss = cm.MSELoss(const=1., reduction='mean')
# The above loss can be described as:
#   loss(x, truth) = 1/l * const * sum(norm2(x, axis=-1), axis=0)

model.compile(optimizer, loss, metrics=[MEE(), RMSE()]
history = model.train(
    train_dataloader, eval_dataloader, n_epochs=100,
    callbacks=[EarlyStopping('Val_MEE', patience=2)]
)
```
The above creates an SGD optimizer and a MSE loss, configures the model
for training with MeanEuclideanError and RootMeanSquaredError as metrics
for both training and validation sets, and trains the model on the given
data for at most 100 epochs and with EarlyStopping strategy monitoring
the MEE on validation set.