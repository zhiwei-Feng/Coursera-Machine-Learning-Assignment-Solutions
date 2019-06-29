import numpy as np
from sigmoid import sigmoid, sigmoid_gradient


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambd):
    Theta1 = np.reshape(nn_params[0:(hidden_layer_size * (input_layer_size + 1))],
                        (hidden_layer_size, (input_layer_size + 1)))  # (25,401)
    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))  # (10,26)
    m = X.shape[0]
    X = np.c_[np.ones(m), X]
    # You need to return the following variables correctly
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    # Part 1 Feedforward the neural network and return the cost in the variable J
    a1 = X

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

    # Part 2 Implement the backpropagation algorithm to compute the gradients
    for i in range(m):
        a1 = X[i]
        z2 = Theta1 @ a1.T
        a2 = sigmoid(z2)
        a2 = np.r_[1, a2]
        z3 = Theta2 @ a2
        a3 = sigmoid(z3)

        delta3 = a3 - Y[i]
        z2 = np.r_[1, z2]
        delta2 = (Theta2.T @ delta3) * sigmoid_gradient(z2)
        delta2 = delta2[1:]

        delta3_new = delta3.reshape(delta3.size, 1)
        a2_new = a2.reshape(a2.size, 1)
        delta2_new = delta2.reshape(delta2.size, 1)
        a1_new = a1.reshape(a1.size, 1)

        Theta2_grad = Theta2_grad + delta3_new @ a2_new.T
        Theta1_grad = Theta1_grad + delta2_new @ a1_new.T

    Theta2_grad = Theta2_grad / m
    Theta1_grad = Theta1_grad / m

    grad = np.r_[Theta1_grad.flatten(), Theta2_grad.flatten()]

    # Part 3: Implement regularization with the cost function and gradients.
    # only regularized when j!=0
    Theta2_grad = Theta2_grad + (lambd / m) * np.c_[np.zeros(num_labels), reg_theta2]
    Theta1_grad = Theta1_grad + (lambd / m) * np.c_[np.zeros(hidden_layer_size), reg_theta1]
    grad = np.r_[Theta1_grad.flatten(), Theta2_grad.flatten()]
    return J, grad
