# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    splitdata.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: uahmed <uahmed@student.hive.fi>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/09/18 10:01:51 by uahmed            #+#    #+#              #
#    Updated: 2024/09/18 11:12:20 by uahmed           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from MLP.load_data import getDataFeatures
from MLP.preprocess import trainValSplit
import pandas as pd
import argparse
import numpy as np

def main():
    '''
    Splits data into train and test sets.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_size', type=float, help='The size of the test set.')
    args = parser.parse_args()
    if args.test_size:
        features, data = getDataFeatures('dataset/data.csv')
        trainSet, testSet, _, _ = trainValSplit(X=data, y=None, validationSize=args.test_size, randomState=4)
        train = pd.DataFrame(trainSet, columns=features)
        test = pd.DataFrame(testSet, columns=features)
        train.to_csv('dataset/training.csv')
        test.to_csv('dataset/test.csv')

if __name__ == '__main__':
    main()
