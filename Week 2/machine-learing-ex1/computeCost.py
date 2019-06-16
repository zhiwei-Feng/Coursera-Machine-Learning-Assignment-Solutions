import numpy as np


def computer_cost(X, y, theta):
    """
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the cost of a particular choice of theta
    %               You should set J to the cost.
    """
    m = y.size
    J = np.sum((X @ theta - y) ** 2 / (2 * m))
    return J
