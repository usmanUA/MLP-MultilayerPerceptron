# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: uahmed <uahmed@student.hive.fi>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/09/13 22:17:26 by uahmed            #+#    #+#              #
#    Updated: 2024/10/01 13:11:38 by uahmed           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from MLP.model import MultilayerPerceptron
from MLP.layers import DenseLayer, BatchNormalizationLayer, ActivationLayer
from MLP.load_data import getDataFeatures
from MLP.preprocess import trainValSplit
from MLP.utils import buildLayers, save_weights, plotAccuracy_Cost
import pandas as pd
import argparse
import numpy as np

def main():
    '''
    Trains the dataset and saves the weights into a file.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, help='Total epochs.')
    parser.add_argument('--loss', type=str, help='Loss function.')
    parser.add_argument('--batch_size', type=int, help='Size of batch for gradient descent.')
    parser.add_argument('--learning_rate', type=float, help='Learning Rate.')
    args = parser.parse_args()

    _, data = getDataFeatures('dataset/training.csv')
    X = data[:, 1:]
    y = data[:, 0]
    X_train, X_val, y_train, y_val = trainValSplit(X, y, validationSize=0.2, randomState=4)
    model = MultilayerPerceptron(['M', 'B'])
    network = model.createNetwork([
                DenseLayer(24),
                BatchNormalizationLayer(),
                ActivationLayer(activation='leakyReLU'),
                DenseLayer(24),
                BatchNormalizationLayer(),
                ActivationLayer(activation='leakyReLU'),
                DenseLayer(2, activation='sigmoid', weight_initializer='glorotUniform')])
    model.fit(network, X_train, X_val, y_train, y_val, args.loss, args.learning_rate, args.batch_size, args.epochs)
    # plotAccuracy_Cost(model.train_loss, model.train_accuracy)
    # plotAccuracy_Cost(model.val_loss, model.val_accuracy)
    #save_weights(sc, './dataset/weights.csv', layers, ['M', 'B'])

if __name__ == '__main__':
    main()
