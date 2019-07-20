import numpy as np
from sigmoid import sigmoid


def predict(Theta1, Theta2, X):
    m = X.shape[0]
    a1 = np.c_[np.ones(m), X]

    z2 = a1 @ Theta1.T
    a2 = sigmoid(z2)
    a2 = np.c_[np.ones(m), a2]

    z3 = a2 @ Theta2.T
    a3 = sigmoid(z3)

    h_theta = np.argmax(a3, axis=1) + 1

    return h_theta
