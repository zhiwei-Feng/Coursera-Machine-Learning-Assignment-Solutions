import numpy as np
from sigmoid import sigmoid


def cost_function(theta, X, y):
    m = y.size
    h = sigmoid(X @ theta)
    J = np.sum(-1 / m * (y * np.log(h) + (1 - y) * np.log(1 - h)))
    grad = (X.T @ (h - y)) / m
    return J, grad

