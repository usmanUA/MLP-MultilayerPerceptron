# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    load_data.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: uahmed <uahmed@student.hive.fi>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/09/13 11:50:05 by uahmed            #+#    #+#              #
#    Updated: 2024/09/13 12:01:49 by uahmed           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from MLP.utils import loadDataset, getData, getFeatures

def getDataFeatures(filename='data.csv'):
    '''
    Parses the data csv file.
    Parameters
    ----------
    filename: Name of the file
    Return
    ------
    features: Features of the dataset.
    data: data.
    '''

    dataset = loadDataset(filename)
    features = getFeatures(dataset)
    data = getData(dataset)

    return features, data
