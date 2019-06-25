import numpy as np
from sigmoid import sigmoid


def predict_one_vs_all(all_theta, X):
    m = X.shape[0]
    X = np.c_[np.ones(m), X]

    h_theta = sigmoid(X @ all_theta.T)
    p = np.argmax(h_theta, axis=1) + 1  # based-0 thus add one
    return p
