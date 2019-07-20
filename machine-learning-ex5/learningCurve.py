import numpy as np
from trainLinearReg import train_linear_reg
from linearRegCostFunction import linear_reg_cost_function


def learning_curve(X, y, Xval, yval, lmd):
    m = X.shape[0]

    error_train = np.zeros(m)
    error_val = np.zeros(m)

    for i in range(1, m + 1):
        X_train = X[:i]
        y_train = y[:i]
        theta = train_linear_reg(X_train, y_train, lmd)
        error_train[i - 1] = linear_reg_cost_function(X_train, y_train, theta, 0)[0]
        error_val[i - 1] = linear_reg_cost_function(Xval, yval, theta, 0)[0]

    return error_train, error_val
