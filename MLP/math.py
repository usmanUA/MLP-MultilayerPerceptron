# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    math.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: uahmed <uahmed@student.hive.fi>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/08/30 21:23:49 by uahmed            #+#    #+#              #
#    Updated: 2024/08/30 22:24:20 by uahmed           ###   ########.fr        #
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

