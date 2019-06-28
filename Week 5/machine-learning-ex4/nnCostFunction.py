import numpy as np
from sigmoid import sigmoid


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambd):
    Theta1 = np.reshape(nn_params[0:(hidden_layer_size * (input_layer_size + 1))],
                        (hidden_layer_size, (input_layer_size + 1)))
    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):], (num_labels, (hidden_layer_size + 1)))

    m = X.shape[0]

    # You need to return the following variables correctly
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    # Part 1 Feedforward the neural network and return the cost in the variable J
    a1 = X
    a1 = np.c_[np.ones(m), a1]

    z2 = a1 @ Theta1.T
    a2 = sigmoid(z2)
    a2 = np.c_[np.ones(m), a2]

    z3 = a2 @ Theta2.T
    a3 = sigmoid(z3)

    Y = np.zeros((m, num_labels))
    for i in range(m):
        Y[i, y[i] - 1] = 1

    J = np.sum((-Y) * np.log(a3) - (1 - Y) * np.log(1 - a3)) / m
    reg_theta1 = Theta1[:, 1:]
    reg_theta2 = Theta2[:, 1:]
    J = J + lambd / (2 * m) * (np.sum(reg_theta1 ** 2) + np.sum(reg_theta2 ** 2))  # regularized cost function
    return J
