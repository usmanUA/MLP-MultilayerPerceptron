# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    preprocess.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: uahmed <uahmed@student.hive.fi>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/09/04 20:02:56 by uahmed            #+#    #+#              #
#    Updated: 2024/09/18 10:25:45 by uahmed           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
from MLP.math import Mean, Std

def trainValSplit(X, y=None, validationSize=0.3, randomState=None):
    '''
    Splits the data into train and test sets.
    Parameters
    ----------
    X: array, shape [n_samples, n_feautures]
    y: array, shape [n_features], default None
    validationSize: float, default 0.3
    randomState: int, default None
    '''

    if randomState:
        np.random.seed(randomState)
    p = np.random.permutation(len(X))
    Xoffset = int(len(X) * validationSize)
    X_train = X[p][Xoffset:]
    X_val = X[p][:Xoffset]
    y_train = None
    y_val = None
    if y is None:
        return X_train, X_val, y_train, y_val
    yOffset = int(len(y) * validationSize)
    y_train = y[p][yOffset:]
    y_val = y[p][:yOffset]
    return X_train, X_val, y_train, y_val

class   Standardizer(object):
    '''
    Scales the X training data using mean and std of the data
    Attributes
    ----------
    _mean: 1D array of size equal to the number of features.
        Mean of the training dataset
    _std: 1D array of size equal to the number of features.
        Standard deviation of the training dataset
    '''

    def __init__(self, mean=np.array([]), std=np.array([])) -> None:
        self.mean = mean
        self.std = std
        self.gamma = None
        self.beta = None

    def __call__(self, X):
        '''
        Applies batch normalization to the inputs.

        Parameters
        ----------
        X: array, shape [n_samples, n_feautures]

        '''
        features = X.shape[1]
        for index in range(features):
            self.mean = np.append(self._mean, Mean(X[:, index]))
            self.std = np.append(self._std, Std(X[:, index]))

    def transform(self, X):
        '''
        Transforms the given data using its mean and std.
        Parameters
        ----------
        X: array, shape [n_samples, n_feautures]
        '''
        return ((X - self._mean) / self._std)
