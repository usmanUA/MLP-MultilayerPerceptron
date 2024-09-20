# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    analyze.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: uahmed <uahmed@student.hive.fi>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/09/08 16:35:06 by uahmed            #+#    #+#              #
#    Updated: 2024/09/18 10:12:02 by uahmed           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import argparse
from MLP.load_data import getDataFeatures
from MLP.describe import describe
from MLP.pairplot import pairplot

def main():
    '''
    Builds a Classifier using the health data.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--describe',type=str, help='describes dataset')
    parser.add_argument('--pairplot',type=str, help='plots pairplot')
    args = parser.parse_args()

    features, data = getDataFeatures('dataset/data.csv')

    if args.describe == 'yes':
        describe(features, data)

    if args.pairplot == 'yes':
        pairplot(features, data)

if __name__ == '__main__':
    main()
