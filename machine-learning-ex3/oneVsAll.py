import numpy as np
import scipy.optimize as op
from lrCostFunction import cost_func, grad_func


def one_vs_all(X, y, num_labels, lmd):
    m, n = X.shape
    X = np.c_[np.ones(m), X]
    all_theta = np.zeros((num_labels, n + 1))

    for i in range(1, num_labels + 1):
        initial_theta = np.zeros(n + 1)
        y_i = np.array([1 if x == i else 0 for x in y])
        res = op.minimize(cost_func, x0=initial_theta, args=(X, y_i, lmd), method='TNC', jac=grad_func)
        all_theta[i - 1] = res.x

    return all_theta
