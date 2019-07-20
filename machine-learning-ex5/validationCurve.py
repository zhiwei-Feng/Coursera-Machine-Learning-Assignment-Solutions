import numpy as np
from trainLinearReg import train_linear_reg
from learningCurve import learning_curve
from linearRegCostFunction import linear_reg_cost_function


def validation_curve(X, y, Xval, yval):
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

    error_train = np.zeros(lambda_vec.size)
    error_val = np.zeros(lambda_vec.size)

    for i in range(lambda_vec.size):
        lmd = lambda_vec[i]
        theta = train_linear_reg(X, y, lmd)

        error_train[i] = linear_reg_cost_function(X, y, theta, 0)[0]
        error_val[i] = linear_reg_cost_function(Xval, yval, theta, 0)[0]

    return lambda_vec, error_train, error_val
