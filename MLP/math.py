# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    math.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: uahmed <uahmed@student.hive.fi>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/08/30 21:23:49 by uahmed            #+#    #+#              #
#    Updated: 2024/09/18 10:28:14 by uahmed           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def Count(X):
    '''Calculates and returns the count of the given feature column.'''
    try:
        X = X.astype("float")
        X = X[~np.isnan(X)]
    except:
        pass
    return len(X)

def Mean(X):
    '''Calculates and returns mean of the given feature column.'''
    tot = 0
    count = 0;
    for val in X:
        if np.isnan(val):
            continue
        count += 1
        tot += val
    return tot / count 

def Std(X):
    '''Calculates and returns standard deviation of the given feature column.'''
    mean = Mean(X)
    tot = 0
    count = 0
    for val in X:
        if np.isnan(val):
            continue
        count += 1
        tot += (val - mean) ** 2
    return (tot / count) ** 0.5


def Min(X):
    '''Calculates and returns min of the given feature column.'''
    min = X[0]
    for val in X:
        if np.isnan(val):
            continue
        if min > val:
            min = val
    return min

def Max(X):
    '''Calculates and returns max of the given feature column.'''
    max = X[0]
    for val in X:
        if np.isnan(val):
            continue
        if max < val:
            max = val
    return max

def Percentile(X, p):
    '''Calculates and returns pth percentile of the given feature column.'''
    k = (len(X)-1) * (p/100)
    f = np.floor(k)
    c = np.ceil(k)
    if f == c:
        return X[int(k)]
    d0 = X[int(f)] * (c - k)
    d1 = X[int(c)] * (f - k)
    return d0 + d1

def HeUniform(Nin, Nout):
    '''
    Initializes the Weights based on HeUniform Distribution.
    '''
    input = 6 / Nin
    W = np.random.uniform(-np.sqrt(input), np.sqrt(input), (Nout, Nin))
    return W

def HeNormal(Nin, Nout):
    '''
    Initializes the Weights based on HeNormal Distribution.
    '''
    input = 2 / Nin
    W = np.random.randn(0, np.sqrt(input), (Nin, Nout))
    return W

def GlorotUniform(Nin, Nout):
    '''
    Initializes the Weights based on GlorotUniform Distribution.
    '''
    input = 6 / (Nin + Nout)
    W = np.random.uniform(-np.sqrt(input), np.sqrt(input), (Nin, Nout))
    return W

def GlorotNormal(Nin, Nout):
    '''
    Initializes the Weights based on GlorotNormal Distribution.
    '''
    input = 2 / (Nin + Nout)
    W = np.random.randn(0, np.sqrt(input), (Nin, Nout))
    return W

def LeCunNormal(Nin, Nout):
    '''
    Initializes the Weights based on LeCunNormal Distribution.
    '''
    W = np.random.randn(0, np.sqrt(1 / Nin), (Nin, Nout))
    return W

def sigmoid(Z):
    '''

    '''
    return 1 / (1 + np.exp(Z))

def ReLU(Z):
    '''

    '''
    zeros = np.zeros([Z.shape[0], Z.shape[1]])
    return max(zeros, Z)

def Tanh(Z):
    '''

    '''
    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

def softmax(A):
    '''

    '''
    num = np.exp(A)
    den = np.exp(A[:, 0]) + np.exp(A[:, 1])
    return num / den

def dsigmoid(Z):
    '''

    '''
    return sigmoid(Z).dot((1 - sigmoid(Z)).T)

def dReLU(Z):
    '''

    '''

    Z[Z > 0] = 1
    Z[Z <= 0] = 0
    return Z

def dTanh(Z):
    '''

    '''
    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

def BinaryCrossEntropy(predY, trueY):
    '''

    '''
    m = trueY.shape[0]
    yOneTerm = trueY.dot(np.log(predY).T)
    yZeroTerm = (1 - trueY).dot(np.log(1 - predY).T)
    return -sum(yOneTerm + yZeroTerm) / m

def CategoricalCrossEntropy(predY, trueY):
    '''

    '''
    m = trueY.shape[0]
    yOneTerm = trueY.dot(np.log(predY).T)
    yZeroTerm = (1 - trueY).dot(np.log(1 - predY).T)
    return sum(yOneTerm + yZeroTerm) / m

def MSE(predY, trueY):
    '''

    '''
    m = trueY.shape[0]
    sqDiff = (trueY - predY) ** 2
    return sum(sqDiff) / m

weight_initializers = {
    'heUniform': HeUniform,
    'heNormal': HeNormal,
    'glorotUniform': GlorotUniform,
    'glorotNormal': GlorotNormal,
    'leCunNormal': LeCunNormal
}

activations = {
    'sigmoid': [sigmoid, dsigmoid],
    'ReLU': [ReLU, dReLU],
    'Tanh': [Tanh, dTanh],
    'softmax': softmax
}

cost_functions = {
    'binaryCrossentropy': BinaryCrossEntropy,
    'categoricalCrossentropy': CategoricalCrossEntropy
}
