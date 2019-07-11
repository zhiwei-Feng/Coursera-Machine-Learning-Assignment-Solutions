import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from findClosestCentroid import find_closest_centroid
from computeCentroids import compute_centroids
from runkMeans import run_kmeans

plt.ion()
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

# ================= Part 1: Find Closest Centroids ====================
print('Finding closest centroids.\n\n')

# Load an example dataset that we will be using
data = sio.loadmat('ex7data2.mat')
X = data['X']

# Select an initial set of centroids
K = 3
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Find the closest centroids for the examples using the
# initial_centroids
idx = find_closest_centroid(X, initial_centroids)

print('Closest centroids for the first 3 examples: \n')
print(' {}'.format(idx[:3]))
print('\n(the closest centroids should be 1, 3, 2 respectively)\n')

input('Program paused. Press enter to continue.\n')

# ===================== Part 2: Compute Means =========================
print('\nComputing centroids means.\n\n')

# Compute means based on the closest centroids found in the previous part.
centroids = compute_centroids(X, idx, K)

print('Centroids computed after initial finding of closest centroids: \n')
print(' {} \n'.format(centroids))
print('\n(the centroids should be\n')
print('   [ 2.428301 3.157924 ]')
print('   [ 5.813503 2.633656 ]')
print('   [ 7.119387 3.616684 ]\n\n')

input('Program paused. Press enter to continue.\n')

# =================== Part 3: K-Means Clustering ======================
print('\nRunning K-Means clustering on example dataset.\n\n')

data = sio.loadmat('ex7data2.mat')
X = data['X']

# Setting for running K-Means
K = 3
max_iters = 10

# here we set centroids to specific values
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Run K-Means algorithm. The 'true' at the end tells our function to plot
# the progress of K-Means
centroids, idx = run_kmeans(X, initial_centroids, max_iters, True)
print('\nK-Means Done.\n\n')

input('Program paused. Press enter to continue.\n')
