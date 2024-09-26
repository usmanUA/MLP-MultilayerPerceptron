# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    BackProp.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: uahmed <uahmed@student.hive.fi>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/09/17 20:07:40 by uahmed            #+#    #+#              #
#    Updated: 2024/09/17 21:38:18 by uahmed           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
from MLP.math import activations

class   BackProp(object):
    '''

    '''

    def __init__(self, eta):
        '''
        Constructs the BackProp object.
        '''
        self._eta = eta

    def propagate(self, mlp, dZ, m):
        '''
        Back propagates and updates the weights.
        Parameters
        ----------
        mlp: feedforward model containing all layers and variables.
        dA: gradient of the loss with respect to the predictions.
        '''
        layerSize = len(mlp.layers) - 1
        while layerSize > 0:
            current = mlp.layers[layerSize]
            prev = mlp.layers[layerSize-1]

            # NOTE: update BN parameters
            current.BN.beta = current.BN.beta - self._eta * np.sum(dZ)
            current.BN.gamma = current.BN.gamma - self._eta * (np.sum(dZ) * current.Z)

            # NOTE: update the weights of current layer
            current.weights = current.weights - self._eta * (1/m) * dZ.dot(prev.A.T)

            # NOTE: update the bias of current layer
            current.bias = current.bias - self._eta * (1/m) * np.sum(dZ, axis=1, keepdims=True)

            # NOTE: compute gradient of the previous layer's activations
            dA_prev = current.weights.T.dot(dZ)

            # NOTE: compute gradient of the previous layer's activation function
            activation_gradient = activations[prev.activation][1](prev.Z)

            # NOTE: compute gradient of the previous layer's logits
            dZ = dA_prev * activation_gradient

            layerSize = layerSize - 1

        # NOTE: update BN parameters for the last layer
        current.BN.beta = current.BN.beta - self._eta * np.sum(dZ)
        current.BN.gamma = current.BN.gamma - self._eta * (np.sum(dZ) * current.Z)

        # NOTE: last layer's weights/bias update
        current = mlp.layers[layerSize]
        current.weights = current.weights - self._eta * (1/m) * dZ.dot(current.X.T)
        current.bias = current.bias - self._eta * (1/m) * np.sum(dZ, axis=1, keepdims=True)
