# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    preprocess.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: uahmed <uahmed@student.hive.fi>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/09/04 20:02:56 by uahmed            #+#    #+#              #
#    Updated: 2024/10/01 13:15:43 by uahmed           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

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

