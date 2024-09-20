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
from MLP.math import cost_functions
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

    def __call__(self, X):
        '''
        Calls each layer and saves the outputs.
        Parameters
        ----------
        layers: list of layers
        '''
        for layer in self.layers:
            X = layer(X)
        return X

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
        self._loss = []
        self._accuracy = []


    def fit(self, X, y):
        '''
        Trains Multilayer Perceptron.
        Parameters
        ----------
        X: training features
        y: true values to learn
        '''
        y_matr = self.makeY(y)
        for _ in range(self._epochs):
            outputs = self._mlp(X)
            self._backprop.propagate(mlp, outputs - y_matr)
            self._cost.append(cost_functions[self._loss](outputs, y_matr))

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
