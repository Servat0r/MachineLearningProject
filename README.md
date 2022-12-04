
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

`Model Creation`
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
train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
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

model.compile(optimizer, loss, metrics=[MEE(), RMSE()])
history = model.train(
    train_dataloader, eval_dataloader, n_epochs=100,
    callbacks=[
        EarlyStopping('Val_MEE', patience=2),
        TrainingCSVLogger()
    ]
)
```
The above creates an SGD optimizer and a MSE loss, configures the model
for training with MeanEuclideanError and RootMeanSquaredError as metrics
for both training and validation sets, trains the model on the given
data for at most 100 epochs and with EarlyStopping strategy monitoring
the MEE on validation set and uses `TrainCSVLogger` for logging training
and validation results at the end of each epoch in a csv file.

`Results and plotting`
```
import matplotlib.pyplot as plt

epochs = np.arange(len(history))
for metric_name, metric_vals in history.items():
    plt.plot(epochs, metric_vals, label=metric_name)
plt.legend()
plt.show()
```
`Model.train()` returns a `core.callbacks.History` object, that contains
a `self.logbook` dictionary attribute of the form: `<metric_name>: <numpy array of
metric values for each epoch>` for each metric. `History` class exposes
also standard `items()`, `keys()` and `values()` methods of `dict` object
for directly iterating through `logbook`. Moreover, `History.__len__()`
provides the number of epochs for which actual data have been registered (that can be lower
than `n_epochs` if e.g. `EarlyStopping` is used in training), and it *must* be used 
to create x-values (epochs) for plotting.

`Model Saving and Loading`
```
model.set_to_eval()
model.save('final_model.model',
    include_compile_objs=True, include_history=True)

# Reload the model from given file
model2 = Model.load('final_model.model')

# Verify that loaded model is equivalent (apart from training updates)
# to previous one
assert model.equal(model2, include_updates=False)
```
The above first sets `model` to `eval` mode, such that backward pass
does *not* compute gradients (since they won't be used when evaluating) and weights/biases
updates will be discarded when saving (e.g., training cycle has terminated), then
saves the model to `"final_model.model"` using pickle and including optimizer, loss,
regularizers and `model.history` in saved data. Then, the model is loaded from
given file into `model2` and the final statement checks that `model` and `model2`
are equal excluding weights/biases updates (by passing `include_updates=False`).

If a complete backup that includes also weights/biases updates and momentum values is needed,
`model.save(serialize_all=True)` serializes also them (still need to pass
`include_compile_objs=True` and `include_history=True` for saving also optimizer,
loss, regularizers and history). In that case, it can be passed `include_updates=True`
to `model.equal()`.

`Metrics and Callbacks`

```
class Metric(Callable):
    
    def update(self, predicted, truth):
        pass
    
    def result(self, batch_num: int = None):
        pass
    
    def reset(self):
        pass
```
Metrics are classes defined in `core.metrics`. The above shows
the base interface of `Metric` objects: the `update()` method is used after each (mini)batch to update the internal state
given model predictions and ground truth values, while `result()` returns the
value of the metric with the following semantic: if `batch_num` is an actual integer,
the result over ***that batch*** is returned, otherwise if `batch_num` is `None` the returned
result is over the ***entire epoch***; `reset()` is used at the end of each epoch. 
Metric names in `History.logbook` are defined by the `Metric.name` attribute, which by default
returns the name of the class corresponding to the metric (e.g. `CategoricalAccuracy`).
For the common metrics `MeanSquaredError`, `MeanEuclideanError` and `RootMeanSquaredError` there
are shortcut variables `MSE`, `MEE` and `RMSE`. By default, for each training metric the name
in `logbook` corresponds to `Metric.name` attribute and for each validation one is `"Val_" + Metric.name`
(e.g. `Val_MEE`).

```
class Callback:
    
    def before_training_cycle(self, model, logs=None):
        pass
    
    def before_training_epoch(self, model, epoch, logs=None):
        pass

    def before_training_batch(self, model, epoch, batch, logs=None):
        pass
    
    def before_evaluate(self, model, epoch=None, logs=None):
        pass

    def before_test_cycle(self, model, logs=None):
        pass

    def before_test_batch(self, model, logs=None):
        pass
```
Callbacks are used to customize behavior of the training/validation/test process.
The above shows some of the `Callback` methods, and in particular for each method
above there exists the correspondent `after_*` version. Each method accepts a `model`
object representing the current model and a `logs` dictionary that contains the recorded
values for that training/validation epoch/minibatch if there are any or `None` otherwise
(e.g. at the start of a training cycle).