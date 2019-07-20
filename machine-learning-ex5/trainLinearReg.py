import numpy as np
from scipy.optimize import fmin_cg
from linearRegCostFunction import linear_reg_cost_function


def train_linear_reg(X, y, lmd):
    init_theta = np.zeros(X.shape[1])

    def cost_function(t):
        return linear_reg_cost_function(X, y, t, lmd)[0]

    def gradient(t):
        return linear_reg_cost_function(X, y, t, lmd)[1]

    theta = fmin_cg(cost_function, x0=init_theta, fprime=gradient, maxiter=200)
    return theta
