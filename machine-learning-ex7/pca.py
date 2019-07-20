import numpy as np
from scipy.linalg import svd


def pca(X):
    m, n = X.shape
    U = np.zeros(n)
    S = np.zeros(n)

    # ====================== YOUR CODE HERE ======================

    # 1. compute Sigma
    Sigma = (1/m)*(X.T @ X)
    # 2. svd
    U, S, V = svd(Sigma)

    # ============================================================

    return U, S, V
