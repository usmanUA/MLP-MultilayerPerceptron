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

from MLP.model import MultilayerPerceptron, MLP
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
    parser.add_argument('--layer', type=int, nargs='+', help='List of number of neurons in hidden layers.')
    parser.add_argument('--activation', type=str, help='Activation function for hidden layers.')
    parser.add_argument('--epochs', type=int, help='Total epochs.')
    parser.add_argument('--loss', type=str, help='Loss function.')
    parser.add_argument('--batch_size', type=int, help='Size of batch for gradient descent.')
    parser.add_argument('--learning_rate', type=float, help='Learning Rate.')
    args = parser.parse_args()

    _, data = getDataFeatures('dataset/training.csv')
    X = data[:, 1:]
    y = data[:, 0]
    X_train, X_val, y_train, y_val = trainValSplit(X, y, validationSize=0.2, randomState=4)
    layers = buildLayers(dims=args.layer, BN=[True, 0.001], activation=args.activation, weight_initializer='heUniform')
    mlp = MLP(layers, args.learning_rate)
    model = MultilayerPerceptron(mlp, args.epochs, args.loss, args.batch_size, ['M', 'B'])
    model.fit(X_train, y_train, X_val, y_val)
    # plotAccuracy_Cost(model.train_loss, model.train_accuracy)
    # plotAccuracy_Cost(model.val_loss, model.val_accuracy)
    #save_weights(sc, './dataset/weights.csv', layers, ['M', 'B'])

if __name__ == '__main__':
    main()
