import numpy as np


def linear_kernel(x1, x2):
    x1 = x1.flatten()
    x2 = x2.flatten()

    return np.vdot(x1, x2)
