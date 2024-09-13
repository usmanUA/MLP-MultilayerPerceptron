# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    scatterplot.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: uahmed <uahmed@student.hive.fi>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/09/01 20:56:41 by uahmed            #+#    #+#              #
#    Updated: 2024/09/02 11:57:51 by uahmed           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import matplotlib.pyplot as plt
from DSLR.utils import getParameters, plotGraph

def scatterplot(features, data):
    '''Plots the Scatter Plot based on the two similar features'''

    xIdx = 6
    yIdx = 8
    _, _, legend, indices = getParameters(features, data)
    data = data[data[:, 0].argsort()]
    X = np.array(data[:,xIdx], dtype=float)
    y = np.array(data[:,yIdx], dtype=float)
    plotGraph(X, y, legend, indices)
    xlabel = features[xIdx]
    ylabel = features[yIdx]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend, loc='upper right', frameon=False)
    plt.savefig('images/scatterplot.png')
    plt.show()
