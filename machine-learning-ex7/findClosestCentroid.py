import numpy as np


def find_closest_centroid(X, centroids):
    idx = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        x_i = X[i]
        norm_2 = np.linalg.norm(x_i-centroids, axis=1)
        idx[i] = np.argmin(norm_2)+1

    return idx
