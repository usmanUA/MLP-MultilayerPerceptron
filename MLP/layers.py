# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    layers.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: uahmed <uahmed@student.hive.fi>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/09/17 09:01:41 by uahmed            #+#    #+#              #
#    Updated: 2024/10/01 13:19:49 by uahmed           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
from MLP.math import weight_initializers, activations, Mean, Std

class DenseLayer(object):
    '''
    Create layers for Multilayer Perceptron.
    '''

    def __init__(self, Nout, BN_info=[False, None], activation='sigmoid', weight_initializer='heUniform'):
        '''
        Constructs a Deep Layer.
        Parameters
        ----------
        shape: size of input Nin and output Nout layers
        activation: The activation function, defualt 'sigmoid'
        weight_initializer: The algorithm to initialize the weights, defualt None
        '''
        self._Nout = Nout
        self.activation = activation
        self._weight_initializer = weight_initializers[weight_initializer]
        self._use_BN = BN_info[0]
        self._epsilon = BN_info[1]
        self._BN = None
        self.X = None
        self.Z = None
        self.A = None
        self._built = False


    def __call__(self, X, Set='train'):
        '''
        Calls the Layers activation.
        Parameters
        ----------
        Z: Linear Outputs

        Return:
        -------
        A: Activation of the layers
        '''
        if not self._built and Set == 'train':
            if self._use_BN == True:
                self._BN = BatchNormalization(epsilon=self._epsilon)
            self.Nin = X.shape[0]
            self.X = X
            self.weights = self._weight_initializer(self.Nin, self._Nout)
            self.bias = np.zeros((self._Nout, 1))
            self._built = True
        if Set == 'train':
            self.Z = self.weights.dot(X) + self.bias
            if self._BN:
                self.A = activations[self.activation][0](self._BN(self.Z))
                return self.A
            self.A = activations[self.activation][0](self.Z)
        elif Set == 'val':
            Z = self.weights.dot(X) + self.bias
            if self._BN:
                Z = self._BN(Z, Set='val')
            return activations[self.activation][0](Z)
        return self.A

    def update(self, dZ, m, eta, prev_layer, last_layer=False):
        '''
        Updates its weights and bias using BackPropagation.

        Parameters
        ----------
        dZ: Gradient of logits received from the following layer.

        Return
        ----------
        dZ: Gradient of logits to pass to the previous layer.
        '''

        if self._BN:
            dZ = self._BN.update(dZ, self.Z, eta, m)

        if last_layer ==  True:
            # NOTE: update the weights of current layer
            self.weights = self.weights - eta * (1/m) * dZ.dot(prev_layer.X.T)

            # NOTE: update the bias of current layer
            self.bias = self.bias - eta * (1/m) * np.sum(dZ, axis=1, keepdims=True)
            return dZ

        # NOTE: update the weights of current layer
        self.weights = self.weights - eta * (1/m) * dZ.dot(prev_layer.A.T)

        # NOTE: update the bias of current layer
        self.bias = self.bias - eta * (1/m) * np.sum(dZ, axis=1, keepdims=True)

        if last_layer ==  True:
            return dZ
        # NOTE: compute gradient of the previous layer's activations
        dA_prev = self.weights.T.dot(dZ) # TODO: make sure whether it is dZ or dY

        # NOTE: compute gradient of the previous layer's activation function
        activation_gradient = activations[prev_layer.activation][1](prev_layer.Z) # TODO: make sure whether it is Z or Y

        # NOTE: compute gradient of the previous layer's logits
        dZ = dA_prev * activation_gradient

        return dZ



class   BatchNormalization(object):
    '''
    Scales the X training data using mean and std of the data
    Attributes
    ----------
    _mean: 1D array of size equal to the number of features.
        Mean of the training dataset
    _std: 1D array of size equal to the number of features.
        Standard deviation of the training dataset
    '''

    def __init__(self, epsilon= 0.001, momentum=0.9, gamma=None, beta=None) -> None:
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon
        self.momentum = momentum
        self._features = None
        self._built = False

    def __call__(self, Z, Set='train'):
        '''
        Applies batch normalization to the inputs.

        Parameters
        ----------
        Z: array, shape [n_samples, n_feautures]

        '''
        if not self._built and Set == 'train':
            self._features = Z.shape[0]
            self.gamma = np.ones((self._features,1))
            self.beta = np.zeros((self._features,1))
            self.mean = np.zeros((self._features,1))
            self.std = np.ones((self._features,1))
            self._built = True
        if Set == 'train':
            mean = np.empty((self._features, 1))
            std = np.empty((self._features, 1))
            for index in range(self._features):
                mean[index, 0] = Mean(Z[index, :])
                std[index, 0] = Std(Z[index, :], mean[index, 0])
            self.Z_norm = (Z - mean) / ((std  + self.epsilon) ** 0.5)
            self.mean = self.momentum * self.mean + (1 - self.momentum) * mean
            self.std = self.momentum * self.std + (1 - self.momentum) * std
            return self.gamma * self.Z_norm + self.beta
        if Set == 'val':
            Z = (Z - self.mean) / ((self.std  + self.epsilon) ** 0.5)
            return self.gamma * Z + self.beta

    def update(self, dY, Z, eta, m):
        '''
        Updates its gamma and beta parameters using BackPropagation.

        Parameters
        ----------
        dZ: Gradient of logits received from the following layer.

        Return
        ----------
        dZ: Gradient of logits to pass to the previous layer.
        '''

        # NOTE: convert dY to dZ
        dZ_norm = dY * self.gamma
        zeroMean = Z - self.mean
        dStd = np.sum(dZ_norm * (zeroMean) * (-0.5) * (self.std + self.epsilon)**(-3/2), axis=0)
        dMean = np.sum(dZ_norm * (-1/np.sqrt(self.std+self.epsilon)), axis=0) + dStd * np.mean(-2*(zeroMean), axis=0)
        dZ = dZ_norm * (1/np.sqrt(self.std+self.epsilon)) + dStd * (2*(zeroMean)/m) + dMean/m

        # NOTE: update BN parameters
        self.beta = self.beta - eta * np.sum(dY, axis=1, keepdims=True)
        self.gamma =self.gamma - eta * np.sum(dY * self.Z_norm, axis=1, keepdims=True)

        return dZ
