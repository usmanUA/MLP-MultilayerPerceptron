# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    preprocessing.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: uahmed <uahmed@student.hive.fi>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/09/04 20:02:56 by uahmed            #+#    #+#              #
#    Updated: 2024/09/04 20:05:47 by uahmed           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
from DSLR.math import Mean, Std

def trainTestSplit(X, y, validationSize=0.3, randomState=None):
    '''
    Splits the data into train and test sets.
    Parameters
    ----------
    X: array, shape [n_samples, n_feautures]
    y: array, shape [n_features]
    validationSize: float, default 0.3
    randomState: int, default None
    '''

    if randomState:
        np.random.seed(randomState)
    p = np.random.permutation(len(X))
    Xoffset = int(len(X) * validationSize)
    yOffset = int(len(y) * validationSize)
    X_train = X[p][Xoffset:]
    X_test = X[p][:Xoffset]
    y_train = y[p][yOffset:]
    y_test = y[p][:yOffset]
    return X_train, X_test, y_train, y_test

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
        self._mean = mean
        self._std = std

    def fit(self, X):
        '''
        Calculates the mean and std of the data
        Parameters
        ----------
        X: array, shape [n_samples, n_feautures]
        '''
        for index in range(X.shape[1]):
            self._mean = np.append(self._mean, Mean(X[:, index]))
            self._std = np.append(self._std, Std(X[:, index]))

    def transform(self, X):
        '''
        Transforms the given data using its mean and std.
        Parameters
        ----------
        X: array, shape [n_samples, n_feautures]
        '''
        return ((X - self._mean) / self._std)
