import numpy as np


def linear_reg_cost_function(X, y, theta, lmd):
    m = y.size

    h_theta = X @ theta
    J = np.sum((h_theta - y) ** 2) / (2 * m) + lmd * np.sum(theta[1:] ** 2) / (2 * m)

    grad = (X.T @ (h_theta - y)) / m + lmd * np.r_[0, theta[1:]] / m

    return J, grad
