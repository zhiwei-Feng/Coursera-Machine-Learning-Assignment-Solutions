from scipy.io import loadmat
import matplotlib.pyplot as plt
from featureNormalize import feature_normalize
from pca import pca
from runkMeans import draw_line
from projectData import project_data
from recoverData import recover_data
from displayData import display_data

plt.ion()
# ================== Part 1: Load Example Dataset  ===================
print('Visualizing example dataset for PCA.\n\n')

data = loadmat('ex7data1.mat')
X = data['X']

# Visualize the example dataset
plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors='b', s=20)
plt.axis('equal')
plt.axis([0.5, 6.5, 2, 8])

input('Program paused. Press enter to continue.\n')

# =============== Part 2: Principal Component Analysis ===============
print('\nRunning PCA on example dataset.\n\n')

# Before running PCA, it is important to first normalize X
X_norm, mu, sigma = feature_normalize(X)

# Run PCA
U, S, V = pca(X_norm)

'''
%  Draw the eigenvectors centered at mean of data. These lines show the
%  directions of maximum variations in the dataset.
'''
plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors='b', s=20)
plt.axis('equal')
plt.axis([0.5, 6.5, 2, 8])
draw_line(mu, mu+1.5*S[0]*U[:, 0])
draw_line(mu, mu+1.5*S[1]*U[:, 1])

print('Top eigenvector: ')
print(' U[:,0] = {} {} '.format(U[0, 0], U[1, 0]))
print('(you should expect to see -0.707107 -0.707107)\n')

input('Program paused. Press enter to continue.\n')

# =================== Part 3: Dimension Reduction ===================
print('\nDimension reduction on example dataset.\n\n')

# Plot the normalized dataset(returned from pca)
plt.scatter(X_norm[:, 0], X_norm[:, 1],
            facecolors='none', edgecolors='b', s=20)
plt.axis('equal')
plt.axis([-4, 3, -4, 3])

# Project the data onto K=1 dimension
K = 1
Z = project_data(X_norm, U, K)
print('Projection of the first example: {}'.format(Z[0]))
print('(this value should be about 1.481274)\n\n')

X_rec = recover_data(Z, U, K)
print('Approximation of the first example: {} {}'.format(
    X_rec[0, 0], X_rec[0, 1]))
print('(this value should be about  -1.047419 -1.047419)')

# Draw lines connecting the projected points to the original points
plt.scatter(X_rec[:, 0], X_rec[:, 1], facecolors='none', edgecolors='r', s=20)
for i in range(X_norm.shape[0]):
    draw_line(X_norm[i, :], X_rec[i, :])
input('Program paused. Press enter to continue.\n')

# =============== Part 4: Loading and Visualizing Face Data =============
print("\nLoading face dataset.\n\n")
data = loadmat('ex7faces.mat')
X = data['X']  # (5000,1024)

# display
display_data(X[:100, :])

input('Program paused. Press enter to continue.\n')

# =========== Part 5: PCA on Face Data: Eigenfaces  ===================
print('Running PCA on face dataset.')
print('(this might take a minute or two ...)\n\n')

'''
%  Before running PCA, it is important to first normalize X by subtracting
%  the mean value from each feature
'''
X_norm, mu, sigma = feature_normalize(X)

# Run PCA
U, S, V = pca(X_norm)

# visualize the top 36 eigenvectors found
display_data(U[:, :36].T)

input('Program paused. Press enter to continue.\n')

# ============= Part 6: Dimension Reduction for Faces =================
print('\nDimension reduction for face dataset.\n\n')

K = 100
Z = project_data(X_norm, U, K)

print('The projected data Z has a size of: ')
print('{} '.format(Z.shape))

input('Program paused. Press enter to continue.\n')

# ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
print('\nVisualizing the projected (reduced dimension) faces.\n\n')

K = 100
X_rec = recover_data(Z, U, K)

# Display normalized data
plt.subplot(1, 2, 1)
display_data(X_norm[:100, :])
plt.title('Original faces')
plt.axis('equal')

# Display reconstructed data from only k eigenfaces
plt.subplot(1, 2, 2)
display_data(X_rec[:100, :])
plt.title('Recovered faces')
plt.axis('equal')

input('Program paused. Press enter to continue.\n')
