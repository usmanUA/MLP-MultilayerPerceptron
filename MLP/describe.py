# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    describe.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: uahmed <uahmed@student.hive.fi>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/09/01 13:14:44 by uahmed            #+#    #+#              #
#    Updated: 2024/09/01 15:09:36 by uahmed           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from MLP.math import *

def describe(features, data):
    '''Mimics pandas dataframe.describe() function'''

    print(f'{"":15}|{"Count  ":>11}|{"Mean  ":>11}|{"Std  ":>11}|{"Min  ":>11}|{"25%  ":>11}|{"50%  ":>11}|{"75%  ":>11}|{"Max  ":>11}')
    for i, feature in enumerate(features):
        print(f"{feature:15.15}", end="|")
        try:
            X = np.array(data[:, i], dtype=float)
            X = X[~np.isnan(X)]
            if not X.any():
                raise Exception('NaN value encountered')
            print(f"{Count(X):>11.4f}", end="|")
            print(f"{Mean(X):>11.4f}", end="|")
            print(f"{Variance(X):>11.4f}", end="|")
            print(f"{Min(X):>11.4f}", end="|")
            print(f"{Percentile(X, 25):>11.4f}", end="|")
            print(f"{Percentile(X, 50):>11.4f}", end="|")
            print(f"{Percentile(X, 75):>11.4f}", end="|")
            print(f"{Max(X):>11.4f}")
        except:
            print(f"{Count(data[:, i]):11.4f}", end="|")
            print(f'{"Categorical feature":>40}')
