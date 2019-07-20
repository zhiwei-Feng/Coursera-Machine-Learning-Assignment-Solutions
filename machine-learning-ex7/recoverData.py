import numpy as np


def recover_data(Z, U, K):
    U_reduce = U[:, :K]
    X_rec = Z @ U_reduce.T
    return X_rec
