# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    layers.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: uahmed <uahmed@student.hive.fi>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/09/17 09:01:41 by uahmed            #+#    #+#              #
#    Updated: 2024/09/17 20:18:34 by uahmed           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
from MLP.math import weight_initializers, activations

class DenseLayer(object):
    '''
    Create layers for Multilayer Perceptron.
    '''

    def __init__(self, Nout, activation='sigmoid', weight_initializer='heUniform'):
        '''
        Constructs a Deep Layer.
        Parameters
        ----------
        shape: size of input Nin and output Nout layers
        activation: The activation function, defualt 'sigmoid'
        weight_initializer: The algorithm to initialize the weights, defualt None
        '''
        self._Nout = Nout
        self._activation = activation
        self._weight_initializer = weight_initializers[weight_initializer]
        self.Z = None
        self.A = None
        self._built = False


    def __call__(self, X):
        '''
        Calls the Layers activation.
        Parameters
        ----------
        Z: Linear Outputs

        Return:
        -------
        A: Activation of the layers
        '''
        if not self._built:
            self.Nin = X.shape[0]
            self.weights = self._weight_initializer(self._Nout, self.Nin)
            self.bias = np.zeros((self._Nout, 1))
            self._built = True
        self.Z = self.weights.dot(X) + self.bias
        self.A = activations[self._activation][0](self.Z)
        return self.A
