import numpy as np


def norml_eqn(X, y):
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    return theta
