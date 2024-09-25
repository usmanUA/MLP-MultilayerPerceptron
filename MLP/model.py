# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    model.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: uahmed <uahmed@student.hive.fi>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/09/13 22:23:36 by uahmed            #+#    #+#              #
#    Updated: 2024/09/13 22:24:14 by uahmed           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from MLP.preprocess import trainValSplit, Standardizer
from MLP.math import cost_functions, softmax
from MLP.BackProp import BackProp
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
            X = layer(X, Set)
        preds = softmax(X)
        return preds

class   MultilayerPerceptron(object):
    '''
    Trains Binary Classifier.
    '''

    def __init__(self, mlp, eta, epochs, loss, batch_size, classes):
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
        self._mlp = mlp
        self._backprop = BackProp(eta)
        self._epochs = epochs
        self._lossFunc = loss
        self._batch_size = batch_size
        self._K = classes
        self.train_loss = []
        self.train_accuracy = []
        self.val_loss = []
        self.val_accuracy = []


    def fit(self, train_X, train_y, val_X, val_y):
        '''
        Trains Multilayer Perceptron.
        Parameters
        ----------
        X: training features
        y: true values to learn
        '''
        train_m = train_X.shape[0]
        batches = int(train_m / self._batch_size)
        val_m = val_X.shape[0]
        train_y_matr = self.makeY(train_y).T
        val_y_matr = self.makeY(val_y).T
        for i in range(self._epochs):
            # NOTE: train the model with mini-batches for each epoch
            for j in range(batches):
                train_y = train_y_matr[:, j:self._batch_size+j]
                train_preds = self._mlp(train_X[j:self._batch_size+j, :])
                train_labels = np.where(train_preds > 0.5, 1, 0)
                batch_m = self._batch_size
                self._backprop.propagate(self._mlp, train_preds - train_y, batch_m)

            # NOTE: predictions based on whole training set
            train_preds = self._mlp(train_X, Set='val')
            train_labels = np.where(train_preds > 0.5, 1, 0)

            # NOTE: compute loss and accuracy for the full training set
            self.train_loss.append(cost_functions[self._lossFunc](train_preds, train_y_matr, train_m))
            self.train_accuracy.append(np.mean(train_labels == train_y_matr) * 100)
            print(f'\033[35mtrain accuracy: {self.train_accuracy[i]}\033[0m')

            # NOTE: predictions based on whole validation set
            val_preds = self._mlp(val_X, Set='val')
            val_labels = np.where(val_preds > 0.5, 1, 0)

            # NOTE: compute loss and accuracy for the full validation set
            self.val_loss.append(cost_functions[self._lossFunc](val_preds, val_y_matr, val_m))
            self.val_accuracy.append(np.mean(val_labels == val_y_matr) * 100)
            print(f'\033[36mvalidation accuracy: {self.val_accuracy[i]}\033[0m')

            # NOTE: print train and validation loss for each epoch
            print(f'\033[38mepoch {i}/{self._epochs} - loss: {self.train_loss[i]} - val_loss: {self.val_loss[i]}\033[0m')
        exit(10)

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
