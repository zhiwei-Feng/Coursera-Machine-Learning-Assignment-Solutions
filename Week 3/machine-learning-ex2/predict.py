import numpy as np

from sigmoid import sigmoid


def predict(theta, x):
    p = x @ theta
    p = sigmoid(p)
    p[p >= 0.5] = 1
    p[p < 0.5] = 0
    return p
