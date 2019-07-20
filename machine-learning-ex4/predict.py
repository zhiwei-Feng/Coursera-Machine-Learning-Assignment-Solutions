import numpy as np
from sigmoid import sigmoid


def predict(Theta1, Theta2, X):
    m = X.shape[0]

    x = np.c_[np.ones(m), X]
    h1 = sigmoid(np.dot(x, Theta1.T))
    h1 = np.c_[np.ones(h1.shape[0]), h1]
    h2 = sigmoid(np.dot(h1, Theta2.T))
    p = np.argmax(h2, axis=1) + 1

    return p
