import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from estimateGaussian import estimate_gaussian
from multivariateGaussian import multivariate_gaussian
from visualizeFit import visualize_fit
from selectThreshold import select_threshold

plt.ion()

# ================== Part 1: Load Example Dataset  ===================
print('Visualizing example dataset for outlier detection.\n\n')

# You should now have the variables X, Xval, yval in your environment
data = loadmat('ex8data1.mat')
# X(307,2)
# must add flatten() ,otherwise will cause big trouble
X, Xval, yval = data['X'], data['Xval'], data['yval'].flatten()

# Visualize the example dataset
plt.scatter(X[:, 0], X[:, 1], marker='x', c='b', s=15)
plt.axis([0, 30, 0, 30])
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')

input('Program paused. Press enter to continue.\n')

# ================== Part 2: Estimate the dataset statistics ===================
print('Visualizing Gaussian fit.\n')
# Estimate mu and sigma2
mu, sigma2 = estimate_gaussian(X)

# Returns the density of the multivariate normal at each data point (row) of X
p = multivariate_gaussian(X, mu, sigma2)

# Visualize the fit
visualize_fit(X, mu, sigma2)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')

input('Program paused. Press enter to continue.\n')

# ================== Part 3: Find Outliers ===================
pval = multivariate_gaussian(Xval, mu, sigma2)

epsilon, F1 = select_threshold(yval, pval)
print('Best epsilon found using cross-validation: {}'.format(epsilon))
print('Best F1 on Cross Validation Set:  {}'.format(F1))
print('   (you should see a value epsilon of about 8.99e-05)')
print('   (you should see a Best F1 value of  0.875000)\n')

# Find the outliers in the training set and plot the
outliers = np.where(p < epsilon)
visualize_fit(X, mu, sigma2)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.scatter(X[outliers, 0], X[outliers, 1], marker='o',
            facecolors='none', edgecolors='r')

input('Program paused. Press ENTER to continue')

# ================== Part 4: Multidimensional Outliers ===================
# X, Xval, yval in your environment
data = loadmat('ex8data2.mat')
X, Xval, yval = data['X'], data['Xval'], data['yval'].flatten()

mu, sigma2 = estimate_gaussian(X)

p = multivariate_gaussian(X, mu, sigma2)

pval = multivariate_gaussian(Xval, mu, sigma2)

epsilon, F1 = select_threshold(yval, pval)
print('Best epsilon found using cross-validation: {}'.format(epsilon))
print('Best F1 on Cross Validation Set:  {}'.format(F1))
print('   (you should see a value epsilon of about 1.38e-18)')
print('   (you should see a Best F1 value of  0.615385)\n')
print('# Outliers found: {}\n\n'.format(np.sum(p < epsilon)))
