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


class   BackProp(object):
    '''

    '''

    def __init__(self, eta):
        '''
        Constructs the BackProp object.
        '''
        self._eta = eta

    def propagate(self, mlp, dZ):
        '''
        Back propagates and updates the weights.
        Parameters
        ----------
        mlp: feedforward model containing all layers and variables.
        dA: gradient of the loss with respect to the predictions.
        '''
        layerSize = size(mlp.layers) - 1
        while layerSize > 0:
            current = mlp.layers[layerSize]
            prev = mlp.layers[layerSize-1]
            current.weights -= self._eta * (1/m) * dZ.dot(prev.A.T)
            current.bias -= self._eta * (1/m) * sum(dZ)
            dA_prev = current.weights.T.dot(dZ)
            activation_gradient = prev.A.dot(1 - prev.A)
            dZ = dA_prev.dot(activation_gradient)
            layerSize = layerSize - 1
