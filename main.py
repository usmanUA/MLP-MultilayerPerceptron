# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    main.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: uahmed <uahmed@student.hive.fi>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/09/08 16:35:06 by uahmed            #+#    #+#              #
#    Updated: 2024/09/08 16:39:03 by uahmed           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import argparse

def main():
    '''
    Builds a Classifier using the health data.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='input dataset')
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    print(df)


if __name__ == '__main__':
    main()
