import numpy as np


def computer_cost(X, y, theta):
    """
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the cost of a particular choice of theta
    %               You should set J to the cost.
    """
    m = len(y)
    h = X @ theta
    y = y.reshape(m, 1)
    J = np.vdot(h - y, h - y) / (2 * m)
    return J
