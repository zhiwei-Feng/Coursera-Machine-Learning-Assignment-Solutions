import numpy as np


def kmeans_init_centroids(X, K):
    #  Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    # Take the first K examples as centroids
    centroids = X[randidx[:K], :]
    return centroids
