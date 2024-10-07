# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    layers.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: uahmed <uahmed@student.hive.fi>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/09/17 09:01:41 by uahmed            #+#    #+#              #
#    Updated: 2024/10/07 09:30:38 by uahmed           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
from MLP.math import weight_initializers, activations, Mean, Variance

class DenseLayer(object):
    '''
    Create layers for Multilayer Perceptron.
    '''

    def __init__(self, Nout, activation=None, weight_initializer='heUniform'):
        '''
        Constructs a Deep Layer.

        Parameters
        ----------
        Nout: size of output layer
        activation: The activation function, defualt 'sigmoid'
        weight_initializer: The algorithm to initialize the weights, defualt heUniform
        '''
        self.learnable = True
        self._Nout = Nout
        self.activation = activation
        self._weight_initializer = weight_initializers[weight_initializer]
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
            self.Nin = X.shape[0]
            self.A = None
            self.param1 = self._weight_initializer(self.Nin, self._Nout)
            self.param2 = np.zeros((self._Nout, 1))
            self._built = True
        if Set == 'train':
            self.X = X
            self.Z = self.param1.dot(self.X) + self.param2
#            print(f'\033[38mDL output: min: {np.min(self.Z)} and max: {np.max(self.Z)} in Z\033[0m')
            if self.activation:
                self.A = activations[self.activation][0](self.Z)
                return self.A
            return self.Z
        elif Set == 'val':
            Z = self.param1.dot(X) + self.param2
            if self.activation:
                return activations[self.activation][0](Z)
            return Z

    def backprop(self, dZ, last_layer=False):#, prev_layer):#, ):
        '''
        Calculates its gradients using BackPropagation.

        Parameters
        ----------
        dZ: Gradient of logits received from the following layer.

        Return
        ----------
        dZ: Gradient of logits to pass to the previous layer.
        '''

        # NOTE: calculate the gradients of weights
        self.grad1 = dZ.dot(self.X.T)

        # NOTE: calculate the gradients of bias
        self.grad2 = np.sum(dZ, axis=1, keepdims=True)

        # NOTE: compute gradient of the previous layer's activations
        dA = self.param1.T.dot(dZ)
        #print(f'\033[38mDLs gradient output: min: {np.min(dA)} and max: {np.max(dA)} in dA\033[0m')
        if not self.activation or (self.activation and last_layer == True):
            return dA

        # NOTE: compute gradient of the previous layer's activation function
        activation_gradient = activations[self.activation][1](self.Z)

        # NOTE: compute gradient of the previous layer's logits
        dZ = dA * activation_gradient

        return dZ



class   BatchNormalizationLayer(object):
    '''
    Scales the X training data using mean and std of the data
    Attributes
    ----------
    _mean: 1D array of size equal to the number of features.
        Mean of the training dataset
    _std: 1D array of size equal to the number of features.
        Standard deviation of the training dataset
    '''

    def __init__(self, epsilon=0.001, momentum=0.99) -> None:
        self.learnable = True
        self.epsilon = epsilon
        self.momentum = momentum
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
            self.param1 = np.ones((self._features,1))
            self.param2 = np.zeros((self._features,1))
            self.mean = np.zeros((self._features,1))
            self.variance = np.ones((self._features,1))
            self._built = True
        if Set == 'train':
            mean = np.mean(Z, axis=1, keepdims=True)
            variance = np.var(Z, axis=1, keepdims=True)
            # mean = np.empty((self._features, 1))
            # variance = np.empty((self._features, 1))
            # for index in range(self._features):
            #     mean[index, 0] = Mean(Z[index, :])
            #     variance[index, 0] = Variance(Z[index, :], mean[index, 0])
            self.Z = Z
            self.Z_norm = (Z - mean) / (np.sqrt(variance  + self.epsilon))
            self.mean = self.momentum * self.mean + (1.0 - self.momentum) * mean
            self.variance = self.momentum * self.variance + (1.0 - self.momentum) * variance
            self.Z_prime = self.param2 * self.Z_norm + self.param2
            #print(f'\033[35mBNs output: min: {np.min(self.Z_prime)} and max: {np.max(self.Z_prime)} in Z_prime\033[0m')
            return self.Z_prime
        if Set == 'val':
            Z = (Z - self.mean) / (np.sqrt(self.variance  + self.epsilon))
            return self.param1 * Z + self.param2

    def backprop(self, dY, m, last_layer=False):
        '''
        Updates its gamma and beta parameters using BackPropagation.

        Parameters
        ----------
        dZ: Gradient of logits received from the following layer.

        Return
        ------
        dZ: Gradient of logits to pass to the previous layer.
        '''

        # NOTE: convert dY to dZ
        dZ_norm = dY * self.param1
        zeroMean = self.Z - self.mean
        invVar = 1.0/np.sqrt(self.variance+self.epsilon)
        dVar = (np.sum(dZ_norm * zeroMean * -0.5 * (self.variance + self.epsilon)**(-3/2), axis=1,keepdims=True))/m
        dMean = (np.sum(dZ_norm*(-1/np.sqrt(self.variance+self.epsilon)),axis=1,keepdims=True)+dVar*(-2 * np.mean(zeroMean,axis=0,keepdims=True)))/m
        dZ = dZ_norm *  + dVar * 2 * zeroMean/m + dMean

        #print(f'\033[35mBNs gradient output: min: {np.min(dZ)} and max: {np.max(dZ)} in dZ\033[0m')
        # NOTE: update BN parameters
        self.grad2 = np.sum(dY, axis=1, keepdims=True)
        self.grad1 = np.sum(dY * self.Z_norm, axis=1, keepdims=True)

        return dZ


class   ActivationLayer(object):
    '''
    Activation layer for applying activation function.
    '''

    def __init__(self, activation):
        '''
        Constructs Activation Layer.

        Parameters
        ---------
        activation: Activation function for activation logits.

        Return
        ------
        Nothing.
        '''
        self.learnable = False
        self.activation = activation

    def __call__(self, Z, Set='train'):
        '''
        Applies Activaion function given the logits.

        Parameters
        ---------
        Z: Activation logits

        Return
        ---------
        A: Activations
        '''

        self.Z = Z
        self.A = activations[self.activation][0](Z)
        #print(f'\033[31mAs output: min: {np.min(self.A)} and max: {np.max(self.A)} in A\033[0m')
        #print(f'A in A: {self.A}')
        return self.A

    def backprop(self, dA, m=None, last_layer=False):
        '''
        Computes logits gradient using BackPropagation.

        Parameters
        ----------
        dA: Gradient of logits received from the following layer.

        Return
        ----------
        dZ: Gradient of logits to pass to the previous layer.
        '''
        self.dZ = dA * activations[self.activation][1](self.Z)
        #print(f'\033[31mAs gradient output: min: {np.min(self.dZ)} and max: {np.max(self.dZ)} in dZ\033[0m')
        return self.dZ
