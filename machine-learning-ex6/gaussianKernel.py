import numpy as np


def gaussian_kernel(x1, x2, sigma):
    x1 = x1.flatten()
    x2 = x2.flatten()

    L2 = np.vdot(x1 - x2, x1 - x2)
    return np.exp(-L2 / (2 * (sigma ** 2)))
