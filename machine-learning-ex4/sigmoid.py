import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))
