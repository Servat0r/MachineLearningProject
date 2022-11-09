import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.decomposition import PCA

DATASET_PATH = "C:/Users/Michele/Documents/MLCUP.csv"

dataset = pd.read_csv(DATASET_PATH)

X = dataset.drop(["output1", "output2", "ID"], axis=1)
# X = X.drop("output2", axis=1)

y = dataset.drop(
    ["ID", "input1", "input2", "input3", "input4", "input5", "input6", "input7", "input8", "input9"], axis=1)

print(X.head())
print(y.head())

# max = np.max(abs(X))
# X = X/max
#
# maxy = np.max(abs(y))
# y = y/maxy
# pca = PCA(n_components=5)
# X = pca.fit_transform(X)
X = (X - np.min(X)) / (np.max(X) - np.min(X))
# yn = (y - np.min(y))/np.ptp(y)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=39)

# max = np.max(abs(X_train))
# X_train = X_train/max
# X_test = X_test/max


if True:
    initializer = tf.keras.initializers.RandomUniform(minval=-0.7, maxval=0.7, seed=11)

    # Create a new model (same as model_2)
    test_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='tanh', kernel_initializer=initializer, kernel_regularizer='l2', kernel_constraint=tf.keras.constraints.MaxNorm(max_value=10.0)),
        tf.keras.layers.Dense(32, activation='tanh', kernel_initializer=initializer, kernel_regularizer='l2', kernel_constraint=tf.keras.constraints.MaxNorm(max_value=10.0)),
        tf.keras.layers.Dense(32, activation='tanh', kernel_initializer=initializer, kernel_regularizer='l2', kernel_constraint=tf.keras.constraints.MaxNorm(max_value=10.0)),
        tf.keras.layers.Dense(2, kernel_regularizer='l2', kernel_initializer=initializer)
    ])

    import keras.backend as K


    def euclidean_distance_loss(y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


    def sum_of_error_loss(y_true, y_pred):
        return K.sum(K.square(y_pred - y_true), axis=-1)


    # Compile the model
    test_model.compile(loss="mse",
                       optimizer=tf.keras.optimizers.SGD(momentum=0.7),
                       metrics=[euclidean_distance_loss, 'mae'])

    early_stopping = EarlyStopping(patience=500, verbose=1, monitor='val_mae')
    checkpoint = ModelCheckpoint("model.h5", monitor='val_mae', mode='min', verbose=1, save_best_only=True)

    # Fit the model
    history = test_model.fit(X_train, y_train, epochs=2000, shuffle=False, batch_size=64,
                             validation_data=(X_test, y_test), callbacks=[checkpoint, early_stopping])

    pd.DataFrame(history.history).plot()
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.ylim([1.3, 2.3])

    plt.show()
    predict = test_model.predict(X_test)
    diff = predict - y_test
    tot = np.sum(abs(diff))
    print(tot)

    # verifica discostamenti
    yVeri1 = np.array(y_test.output1)
    yPredetti1 = predict.transpose()[0]
    plt.scatter(yVeri1, yPredetti1)
    plt.show()

    # verifica discostamenti
    yVeri2 = np.array(y_test.output2)
    yPredetti2 = predict.transpose()[1]
    plt.scatter(yVeri2, yPredetti2)
    plt.show()
