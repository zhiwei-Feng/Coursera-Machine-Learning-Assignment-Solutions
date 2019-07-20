import numpy as np
from sigmoid import sigmoid
from costFunction import cost_function


def cost_function_reg(theta, X, y, lamb):
    m = y.size
    J, grad = cost_function(theta, X, y)
    theta_ex0 = theta[1:]
    J += lamb * np.sum(theta_ex0 ** 2) / (2 * m)
    grad += np.r_[0, theta_ex0] * (lamb / m)
    return J, grad
