# Multilayer Perceptron (MLP) Project

This project implements a **Multilayer Perceptron (MLP)**, a type of feedforward artificial neural network. It is built for the 42 School curriculum and uses Python for creating and training the model.

## Project Overview

An MLP consists of multiple layers of neurons where each neuron in a layer is fully connected to the neurons of the previous layer. This project includes:
- Implementation of a basic MLP from scratch.
- Forward pass and backpropagation algorithms.
- Model training for supervised learning tasks (e.g., classification).

## Features

- Input layer, one or more hidden layers, and an output layer.
- Sigmoid activation function for non-linearity.
- Gradient-based optimization using backpropagation.
- Cross-entropy loss for classification problems.

## Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/mlp-project.git
train.py [-h] [--layer LAYER [LAYER ...]] [--activation ACTIVATION] [--epochs EPOCHS] [--loss LOSS] [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE]
