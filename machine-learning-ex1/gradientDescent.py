import numpy as np

import computeCost


def gradient_descent(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        """
        % ====================== YOUR CODE HERE ======================
        % Instructions: Perform a single gradient step on the parameter vector
        %               theta. 
        %
        % Hint: While debugging, it can be useful to print out the values
        %       of the cost function (computeCost) and gradient here.
        %
        """
        h = (X @ theta).flatten()  # from (97,1) to (97,)
        theta -= alpha * (1 / m) * (X.T @ (h - y))

        # Save the cost J in every iteration
        J_history[i] = computeCost.computer_cost(X, y, theta)

    return theta
