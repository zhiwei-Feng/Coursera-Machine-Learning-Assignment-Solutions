import numpy as np


def poly_features(X, p):
    X_poly = np.zeros((X.shape[0], p))
    for i in range(1, p + 1):
        X_poly[:, i - 1] = np.power(X, i).flatten()

    return X_poly
