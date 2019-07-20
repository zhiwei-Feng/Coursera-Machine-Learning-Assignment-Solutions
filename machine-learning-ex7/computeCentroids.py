import numpy as np


def compute_centroids(X, idx, K):
    _, n = X.shape
    centroids = np.zeros((K, n))

    for k in range(1, K+1):
        ids = np.where(idx == k)
        centroids[k-1] = np.mean(X[ids], axis=0)

    return centroids
