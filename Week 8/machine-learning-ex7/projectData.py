import numpy as np


def project_data(X, U, K):

    # ====================== YOUR CODE HERE ======================
    U_reduce = U[:, :K]
    Z = X @ U_reduce

    return Z
