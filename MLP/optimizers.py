# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    optimizers.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: uahmed <uahmed@student.hive.fi>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/10/07 08:06:32 by uahmed            #+#    #+#              #
#    Updated: 2024/10/07 10:24:28 by uahmed           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

class AdamOptimizer(object):
    '''

    '''

    def __init__(self, learning_rate, epsilon, beta1, beta2, layers):
        '''

        '''
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.layers = layers
        self.m_t = {}
        self.v_t = {}
        self.time = 0
        self.built = False



    def apply_gradients(self, grads):
        '''

        '''
        if not self.built:
            for layer in self.layers:
                if layer.learnable:
                    self.m_t[layer] = {'grad1': np.zeros_like(layer.param1), 'grad2': np.zeros_like(layer.param2)}
                    self.v_t[layer] = {'grad1': np.zeros_like(layer.param1), 'grad2': np.zeros_like(layer.param2)}
            self.built = True
        self.time += 1
        for layer in self.layers:
            if layer.learnable:
                # NOTE: update the variables
                self.m_t[layer]['grad1'] = self.beta1 * self.m_t[layer]['grad1'] + (1.0 - self.beta1) * layer.grad1
                self.m_t[layer]['grad2'] = self.beta1 * self.m_t[layer]['grad2'] + (1.0 - self.beta1) * layer.grad2
                self.v_t[layer]['grad1'] = self.beta2 * self.v_t[layer]['grad1'] + (1.0 - self.beta2) * (layer.grad1 **2)
                self.v_t[layer]['grad2'] = self.beta2 * self.v_t[layer]['grad2'] + (1.0 - self.beta2) * (layer.grad2 **2)

                # NOTE: calculate the variables
                m_t_param1 = self.m_t[layer]['grad1'] / (1 - self.beta1 ** self.time)
                m_t_param2 = self.m_t[layer]['grad2'] / (1 - self.beta1 ** self.time)
                v_t_param1 = self.v_t[layer]['grad1'] / (1 - self.beta2 ** self.time)
                v_t_param2 = self.v_t[layer]['grad2'] / (1 - self.beta2 ** self.time)

                # NOTE: update the learnable parameters
                layer.param1 -= self.learning_rate * m_t_param1 / (np.sqrt(v_t_param1) + self.epsilon)
                layer.param2 -= self.learning_rate * m_t_param2 / (np.sqrt(v_t_param2) + self.epsilon)



class SGD(object):
    '''

    '''

    def __init__(self, learning_rate, layers):
        '''

        '''
        self.learning_rate = learning_rate
        self.layers = layers

    def apply_gradients(self, m):
        '''

        '''
        for layer in self.layers:
            if layer.learnable:
                layer.param1 -= self.learning_rate * (1/m) * layer.grad1
                layer.param2 -= self.learning_rate * (1/m) * layer.grad2
