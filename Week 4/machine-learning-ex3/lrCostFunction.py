import numpy as np
from sigmoid import sigmoid


# ËðÊ§º¯Êý
def lrCostFunc(theta, x, y, lam):
    m = np.size(y, 0)
    h = sigmoid(x.dot(theta))
    j = -1 / m * (y.dot(np.log(h)) + (1 - y).dot(np.log(1 - h))) + lam * (theta[1:].dot(theta[1:])) / (2 * m)
    return j


# ÌÝ¶Èº¯Êý
def lrGradFunc(theta, x, y, lam):
    m = np.size(y, 0)
    h = sigmoid(x.dot(theta))
    grad = np.zeros(np.size(theta))
    grad[0] = 1 / m * (x[:, 0].dot(h - y))
    grad[1:] = 1 / m * (x[:, 1:].T.dot(h - y)) + lam / m * theta[1:]
    return grad
