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

def Std(X, mean):
    '''Calculates and returns variance of the given feature column.'''
    tot = 0.0
    count = 0
    for val in X:
        if np.isnan(val):
            continue
        count += 1
        tot += (val - mean) ** 2
    return (tot / count)


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

    Parameters
    ----------
    Nin: Input dimension
    Nout: Output dimension

    Return
    ------
    W: Randomly initialized weights, (Nin, Nout)
    '''
    limit = np.sqrt(6 / Nin)
    W = np.random.uniform(-limit, limit, (Nout, Nin))
    return W

def HeNormal(Nin, Nout):
    '''
    Initializes the Weights based on HeNormal Distribution.
    '''
    input = 2 / Nin
    W = np.random.randn(0, np.sqrt(input), (Nout, Nin))
    return W

def GlorotUniform(Nin, Nout):
    '''
    Initializes the Weights based on GlorotUniform Distribution.
    '''
    input = np.sqrt(6 / (Nin + Nout))
    W = np.random.uniform(-input, input, (Nout, Nin))
    return W

def GlorotNormal(Nin, Nout):
    '''
    Initializes the Weights based on GlorotNormal Distribution.
    '''
    input = np.sqrt(2 / (Nin + Nout))
    W = np.random.randn(0, input, (Nout, Nin))
    return W

def LeCunNormal(Nin, Nout):
    '''
    Initializes the Weights based on LeCunNormal Distribution.
    '''
    W = np.random.randn(0, np.sqrt(1 / Nin), (Nout, Nin))
    return W

def LeCunUniform(Nin, Nout):
    '''
    Initializes the Weights based on LeCunNormal Distribution.
    '''
    limit = np.sqrt(3 / Nin)
    W = np.random.uniform(-limit, limit, (Nout, Nin))
    return W

def sigmoid(Z):
    '''

    '''
    Z = Z.astype(float)
    return 1.0 / (1.0 + np.exp(-Z))

def ReLU(Z):
    '''
    Computes activation using ReLU activation function.

    Parameter
    ---------
    Z: Input data (n_sampes, n_features)

    Return
    ------
    A: ReLU activation (n_samples, n_features)
    '''
    zeros = np.zeros_like(Z)
    return np.maximum(zeros, Z)

def LeakyReLU(Z, alpha=0.01):
    '''
    Computes activation using Leaky ReLU activation function.

    Parameter
    ---------
    Z: Input data (n_sampes, n_features)
    alpha: default 0.01

    Return
    ------
    A: Leaky ReLU activation (n_samples, n_features)
    '''

    return np.where(Z > 0, Z, Z * alpha)

def Tanh(Z):
    '''
    Computes activation using Tanh activation function.

    Parameter
    ---------
    Z: Input data (n_sampes, n_features)
    alpha: default 0.01

    Return
    ------
    A: Tanh activation (n_samples, n_features)
    '''
    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

def softmax(A):
    '''
    Implements softmax.

    Parameters
    ----------
    A: Activations (n_samples, n_classes)

    Return
    ------
    p: probabilities for each class (n_samples, n_classes)
    '''
    num = np.exp(A - np.max(A, axis=0, keepdims=True))
    den = np.sum(num, axis=0, keepdims=True)
    return num / den

def dsigmoid(A):
    '''
    Computes the gradient of sigmoid function.

    Parameters
    ----------
    A: activation of the current layer

    Return
    ------
    g': gradient of the sigmoid function
    '''
    return A * (1 - A)

def dReLU(Z):
    '''
    Computes the gradient of ReLU function.

    Parameters
    ----------
    A: activation of the current layer

    Return
    ------
    g': gradient of the ReLU function
    '''
    return np.where(Z > 0, 1, 0)

def dLeakyReLU(Z, alpha=0.01):
    '''
    Computes devivative of LeakyReLU activation.

    Parameter
    ---------
    Z: Input data (n_sampes, n_features)
    alpha: default 0.01

    Return
    ------
    dA: derivative of Leaky ReLU activation (n_samples, n_features)
    '''
    return np.where(Z > 0, 1, alpha)

def dTanh(Z):
    '''
    Computes the gradient of Tanh function.

    Parameters
    ----------
    A: activation of the current layer

    Return
    ------
    g': gradient of the Tanh function
    '''
    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

def BinaryCrossEntropy(predY, trueY, m):
    '''
    Calculates the binary cross entropy.

    Parameters
    ---------
    predY: Predicted y matrix (n_classes, n_samples)
    trueY: True y matrix (n_classes, n_samples)

    Return
    ------
    loss: Loss of the predictions
    '''
    yOneTerm = trueY * (np.log(predY))
    yZeroTerm = (1 - trueY) * (np.log(1 - predY))
    loss = (-1 / m) * np.sum(yOneTerm + yZeroTerm)
    return loss

def CategoricalCrossEntropy(predY, trueY):
    '''
    Calculates the categorical cross entropy.

    Parameters
    ---------
    predY: Predicted y matrix (n_classes, n_samples)
    trueY: True y matrix (n_classes, n_samples)

    Return
    ------
    loss: Loss for the model
    '''
    m = trueY.shape[0]
    yOneTerm = trueY.dot(np.log(predY).T)
    yZeroTerm = (1 - trueY).dot(np.log(1 - predY).T)
    return sum(yOneTerm + yZeroTerm) / m

def MSE(predY, trueY):
    '''
    Calculates the mean square error.

    Parameters
    ---------
    predY: Predicted y matrix (n_classes, n_samples)
    trueY: True y matrix (n_classes, n_samples)

    Return
    ------
    loss: Loss for the model
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
    'leakyReLU': [LeakyReLU, dLeakyReLU],
    'Tanh': [Tanh, dTanh],
}

cost_functions = {
    'binaryCrossentropy': BinaryCrossEntropy,
    'categoricalCrossentropy': CategoricalCrossEntropy
}
