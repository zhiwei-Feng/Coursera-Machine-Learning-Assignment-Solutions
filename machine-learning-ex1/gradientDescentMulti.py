import numpy as np

import computeCostMulti


def gradient_descent_multi(X, y, theta, alpha, num_iters):
    """
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    """
    m = y.size
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        h = (X @ theta).flatten()
        theta -= alpha * (1 / m) * (X.T @ (h - y))

        J_history[i] = computeCostMulti.computer_cost(X, y, theta)

    return theta, J_history
