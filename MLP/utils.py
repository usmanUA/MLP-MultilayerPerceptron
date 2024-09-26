# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    utils.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: uahmed <uahmed@student.hive.fi>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/09/07 22:37:31 by uahmed            #+#    #+#              #
#    Updated: 2024/09/13 22:17:46 by uahmed           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from MLP.layers import DenseLayer


def loadDataset(fileName):
    '''Loads the dataset file and writes entries to a numpy array'''
    dataset = []
    with open(fileName, 'r') as file:
        reader = csv.reader(file)
        try:
            for rw in reader:
                row = []
                for entry in rw:
                    try:
                        entry = float(entry)
                    except:
                        if not entry:
                            entry = np.nan
                    row.append(entry)
                dataset.append(row)
        except csv.accuracy as e:
            print(f"File: {fileName}, line number: {reader.line_num}, accuracy: {e}")
    return  np.array(dataset, dtype=object)

def modelData(dataset, features, action):
    '''
    Parses data selecting the features for the training/predictions.
    Parameters
    ----------
    dataset: numpy ndarray of the given dataset

    Returns
    ------
    X: selected features
    y: feature to learn/predict
    '''
    df = pd.DataFrame(dataset, columns=features)
    if action == 'train':
        df = df.dropna(subset=importantFeatures)
    else:
        df = df.fillna(method='ffill')
    y = df.values[:, 0]
    dfX = df[importantFeatures]
    X = dfX.to_numpy(dtype=float)
    return X, y

def getFeatures(dataset):
    ''' Returns the features names of the given dataset (column names)'''
    return dataset[0, 1:]

def getData(dataset):
    ''' Returns the data of the given dataset (excluding the columns names)'''
    return dataset[1:, 1:]

def getLegendsSortData(data, index):
    '''Returns the legends for the plot'''

    sortedData = data[data[:, index].argsort()]
    diagnosis = sortedData[:, index]
    legend = sorted(set(diagnosis))
    return legend, diagnosis, sortedData

def getParameters(features, data):
    '''Parses parameters for histogram plot'''
    title = features[15]
    legend, diagnosis, sortedData = getLegendsSortData(data, 0)
    elems, inds = np.unique(diagnosis, return_index=True)
    indices = {}
    i = 0
    for ind, elem in enumerate(elems):
        indices[legend[i]] = int(inds[ind])
        i += 1
    X = sortedData[:, 15]
    X = X.astype("float")
    X = X[~np.isnan(X)]
    return X, title, legend, indices, sortedData

def plotGraph(X, Y, legend, indices, ax=None):
    '''Plots the graph based on the given instruction in the parameters'''

    tot = len(legend)
    colors = ['red', 'green']
    for i in range(0, tot):
        color = colors[i]
        if i == tot - 1:
            if Y is None:
                h = X[indices[legend[i]]:]
                h = h[~np.isnan(h)]
                if ax is None:
                    plt.hist(h, color=color, alpha=0.5)
                else:
                    ax.hist(h, alpha=0.5)
            else:
                x = X[indices[legend[i]]:]
                y = Y[indices[legend[i]]:]
                if ax is None:
                    plt.scatter(x, y, color=color, alpha=0.5)
                else: 
                    ax.scatter(x, y, s=1, color=color, alpha=0.5)
        else:
            if Y is None:
                h = X[indices[legend[i]]:indices[legend[i+1]]]
                h = h[~np.isnan(h)]
                if ax is None:
                    plt.hist(h, color=color, alpha=0.5)
                else: 
                    ax.hist(h, alpha=0.5)
            else:
                x = X[indices[legend[i]]:indices[legend[i+1]]]
                y = Y[indices[legend[i]]:indices[legend[i+1]]]
                if ax is None:
                    plt.scatter(x, y, color=color, alpha=0.5)
                else:
                    ax.scatter(x, y, s=1, color=color, alpha=0.5)


def plotAccuracy_Cost(cost, accuracy):
    '''
    Plots the cost and accuracy of the Logistic Regression Model.
    Parameters
    ---------
    cost: Cost History, shape (n_iter, )
    accuracy: accuracy History , shape (n_iter, )
    '''

    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), constrained_layout=True)
    ax[0].plot(range(1, len(cost)+1), cost, marker='o', color='violet')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Cost Function')
    ax[0].set_title('Multilayer Preceptron - Learning Rate 0.01')

    ax[1].plot(range(1, len(accuracy)+1), accuracy, marker='o', color='red')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[0].set_title('Multilayer Preceptron - Learning Rate 0.01')

    plt.show()

def buildLayers(dims=[], activation='sigmoid', weight_initializer=None):
    '''
    Builts the layers.
    Parameters
    ----------
    dims: dimension of each layers outputs.
    activation: activation function.
    weight_initializer: activation function.

    Return
    ------
    Layers: List of layer objects

    Return
    ------
    Layers: List of layer objects.
    '''

    layers = []
    tot = len(dims)
    for i in range(tot):
        layers.append(DenseLayer(Nout=dims[i], activation=activation, weight_initializer=weight_initializer))
    layers.append(DenseLayer(Nout=2, activation='sigmoid', weight_initializer='glorotUniform'))
    return layers

def save_weights(sc, filename, layers, classes):
    '''
    Saves the optimized weights to a file.
    Parameters
    ----------
    sc: Standardizer object to calculate mean, std etc.
    filename: Name of the file to save the weights to.
    
    Returns
    -------
    self: Returns the object
    '''

    tot_layers = len(layers)
    with open(filename, 'w') as f:
        for i in range(len(classes)):
            f.write(f'{classes[i]},')
        f.write('Mean,Std\n')
        for i, layer in enumerate(layers):
            for j in range(0, layer.weights.shape[1]):
                for i in range(0, layer.weights.shape[0]):
                    f.write(f'{layer.weights[i][j]},')
                f.write(f'{sc._mean[j - 1]},{sc._std[j - 1]}\n')
