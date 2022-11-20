# Model class
import Optimizers
from Layers import Input
from LossFunctions import LossBase
from Optimizers import OptimezerBase


class Model:
    def __init__(self):
        # Create a list of network objects
        self.trainable_layers = None
        self.input_layer = None
        self.optimizer = None
        self.loss = None
        self.layers = []

    # Add objects to the model
    def add(self, layer):
        self.layers.append(layer)


    def compile(self,*, lossname, optimizer):
        self.loss = LossBase.GetLossByName(lossname)
        self.optimizer = optimizer
        self.input_layer = Input()
        layer_count = len(self.layers)
        #self.trainable_layers = []
        assert (layer_count > 1), "It's not a deep network"

        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss

            # if hasattr(self.layers[i], 'weights'):
            #     self.trainable_layers.append(self.layers[i])



    def train(self, X, y, *, epochs=1, batch_size=None, log_every=1):

        train_steps = 1
        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1

        for epoch in range(1, epochs + 1):
            self.loss.new_pass()
            for step in range(train_steps):
                # If batch size is not set -
                # train using one step and full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step * batch_size:(step + 1) * batch_size]
                    batch_y = y[step * batch_size:(step + 1) * batch_size]

                output = self.forward(batch_X, training = True)

                data_loss,regularization_loss= self.loss.calculate(output,batch_y,self.layers)
                '''
                data_loss = self.loss.forward(output, batch_y)

                regularization_loss = 0
                for layer in self.layers:
                    if layer.isTrainable():
                        regularization_loss += layer.regularization_loss()
                '''

                loss = data_loss + regularization_loss
                predictions = output

                # Perform backward pass
                self.backward(output, batch_y)

                # Optimize (update parameters)
                self.optimizer.pre_update_params()
                for layer in self.layers:
                    if layer.isTrainable():
                        self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

            epoch_data_loss, epoch_regularization_loss =  self.loss.calculate_accumulated(self.layers)
            epoch_loss = epoch_data_loss + epoch_regularization_loss

            if not epoch % log_every:
                print(f'epoch: {epoch}, ' +
                      f'loss: {epoch_loss:.8f} (' +
                      f'data_loss: {epoch_data_loss:.4f}, ' +
                      f'reg_loss: {epoch_regularization_loss:.4f}), ' +
                      f'lr: {self.optimizer.current_learning_rate}')



    def forward(self, X, training):
        self.input_layer.forward(X,training)
        for layer in self.layers:
            layer.forward(layer.prev.output,training)
        return layer.output

    def backward(self, output, y):
        # First call backward method on the loss
        # this will set dinputs property that the last
        # layer will try to access shortly
        self.loss.backward(output, y)

        # Call backward method going through all the objects
        # in reversed order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
