# Multilayer Perceptron (MLP) Project

This project implements a **Multilayer Perceptron (MLP)**, a type of feedforward artificial neural network. It is built for the 42 School curriculum and uses Python for creating and training the model.

## Project Overview

An MLP consists of multiple layers of neurons where each neuron in a layer is fully connected to the neurons of the previous layer. This project includes:
- Implementation of a basic MLP from scratch.
- Forward pass and backpropagation algorithms.
- Model training for supervised learning tasks (e.g., classification).

## Features

- Input layer, one or more hidden layers, and an output layer.
- Activation Functions: Sigmoid, ReLU, Leaky ReLU, Tanh.
- Mini-Batch Gradient-based optimization using backpropagation.
- Loss functions: BinaryCrossEntropy, CategoritcalCrossEntropy, and MeanAbsoluteError.

## Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/mlp-project.git
```
Analyze the data:
```bash
python3 analyze.py -h
python3 analyze.py [-h] [--describe DESCRIBE] [--pairplot PAIRPLOT]
```
Split data into training and test sets:
```bash
python3 splitdata.py -h
python3 splitdata.py [-h] [--test_size TEST_SIZE]
```
Train the model:
```bash
python3 train.py -h
python3 train.py [-h] [--layer LAYER [LAYER ...]] [--activation ACTIVATION] [--epochs EPOCHS] [--loss LOSS] [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE]
```
