# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    model.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: uahmed <uahmed@student.hive.fi>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/09/13 22:23:36 by uahmed            #+#    #+#              #
#    Updated: 2024/10/07 09:51:45 by uahmed           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from MLP.preprocess import trainValSplit
from MLP.math import cost_functions, softmax
import numpy as np

class   MLP(object):
    '''
    Feed Forward Calculator.
    '''

    def __init__(self, layers):
        '''
        Constructs MLP, a feedforward model.
        Parameters
        ----------
        layers: List of DenseLayer objects
        '''
        self.layers = layers

    def __call__(self, X, Set='train'):
        '''
        Calls each layer and saves the outputs.

        Parameters
        ----------
        layers: list of layers

        Return
        ----------
        predictions: prediction probabilities
        '''

        X = X.T
        for layer in self.layers:
            X = np.array(X, dtype=float)
            X = layer(X, Set)
        preds = softmax(X)
        return preds

    def propagate(self, dA, m):
        '''
        Back propagates and updates the weights.
        Parameters
        ----------
        mlp: feedforward model containing all layers and variables.
        dA: gradient of the loss with respect to the predictions.
        '''
        layers = len(self.layers) - 1
        dA = self.layers[layers].backprop(dA, last_layer=True)#, self.layers[layers-1])
        layers = layers - 1
        while layers >= 0:
            dA = self.layers[layers].backprop(dA, m)#, self.layers[layers-1])
            layers = layers - 1


class   MultilayerPerceptron(object):
    '''
    Trains Binary Classifier.
    '''

    def __init__(self, classes):
        '''
        Constucts MultilayerPerceptron.
        Parameters
        ----------
        mlp: Feedforward model containing layers
        eta: Learning rate
        epochs: Number of iterations for gradient descent
        loss: Loss function
        batch_size: Size of each batch
        '''
        self._K = classes
        self.train_loss = []
        self.train_accuracy = []
        self.val_loss = []
        self.val_accuracy = []

    def createNetwork(self, layers):
        '''
        Creates network based on the layers provided.

        Parameters
        ----------
        layers: A list of NN layers

        Return
        ------
        network: Neural Network of all the provided layers.
        '''

        return MLP(layers)


    def fit(self, network, train_X, val_X, train_y, val_y, loss, batch_size, epochs, optimizer):
        '''
        Trains Multilayer Perceptron.

        Parameters
        ----------
        network: Neural Network consisting all layers
        train_data: Training data
        val_data: training features
        loss: Loss Function
        learning_rate: Learning rate
        batch_size: Size of a batch for mini-batch gradient Descent
        epoch: Number of epochs for training the model
        '''
        train_m = train_X.shape[0]
        batches = int(train_m / batch_size)
        val_m = val_X.shape[0]
        train_y_matr = self.makeY(train_y).T
        val_y_matr = self.makeY(val_y).T
        for i in range(epochs):
            # NOTE: train the model with mini-batches for each epoch
            for j in range(batches):
                #print(f'\033[31mBatch Number: {j}\033[0m')
                train_y = train_y_matr[:, j:batch_size+j]
                train_preds = network(train_X[j:batch_size+j, :])
                train_labels = np.where(train_preds > 0.5, 1, 0)
                batch_m = batch_size
                network.propagate(train_preds - train_y, batch_m)
                optimizer.apply_gradients(batch_m)

            # NOTE: predictions based on whole training set
            #print(f'\033[31mPredicting Whole training DATA: {j}\033[0m')
            train_preds = network(train_X, Set='val')
            train_labels = np.where(train_preds > 0.5, 1, 0)

            # NOTE: compute loss and accuracy for the full training set
            self.train_loss.append(cost_functions[loss](train_preds, train_y_matr, train_m))
            self.train_accuracy.append(np.mean(train_labels == train_y_matr) * 100)
            print(f'\033[35mtrain accuracy: {self.train_accuracy[i]}\033[0m')

            # NOTE: predictions based on whole validation set
            #print("Done with Training data, predicting Validation")
            val_preds = network(val_X, Set='val')
            val_labels = np.where(val_preds > 0.5, 1, 0)

            # NOTE: compute loss and accuracy for the full validation set
            self.val_loss.append(cost_functions[loss](val_preds, val_y_matr, val_m))
            self.val_accuracy.append(np.mean(val_labels == val_y_matr) * 100)
            print(f'\033[36mvalidation accuracy: {self.val_accuracy[i]}\033[0m')

            # NOTE: print train and validation loss for each epoch
            print(f'\033[38mepoch {i}/{epochs} - loss: {self.train_loss[i]} - val_loss: {self.val_loss[i]}\033[0m')

        return self

    def makeY(self, y):
        '''
        Builds Y vector based on number of categories for predictions.
        Returns
        -------
        Y: New matrix based on y and number of categories.
        '''
        y_size = len(y)
        y_matr = np.zeros((y_size, len(self._K)))
        for i in range(y_size):
            y_matr[i, self._K.index(y[i])] = 1
        return y_matr
