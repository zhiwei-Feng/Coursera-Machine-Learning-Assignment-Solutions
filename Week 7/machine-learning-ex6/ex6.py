import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from plotData import plot_data
from sklearn import svm
from visualizeBoundaryLinear import visualize_boundary_linear
from visualizeBoundary import visualize_boundary
from gaussianKernel import gaussian_kernel
from dataset3Params import dataset_3_params

# =============== Part 1: Loading and Visualizing Data ================
print('Loading and Visualizing Data ...\n')

# Load from ex6data1:
# You will have X, y in your environment
data = sio.loadmat('ex6data1.mat')
X, y = data['X'], data['y'].flatten()

plot_data(X, y)
plt.show()

input('Program paused. Press enter to continue.\n')

# ==================== Part 2: Training Linear SVM ====================
print('\nTraining Linear SVM ...\n')

'''
You should try to change the C value below and see how the decision
boundary varies (e.g., try C = 1000)
'''
C = 1
model = svm.SVC(C, kernel='linear', tol=1e-3)
model.fit(X, y)
visualize_boundary_linear(X, y, model)
plt.show()

input('Program paused. Press enter to continue.\n')

# =============== Part 3: Implementing Gaussian Kernel ===============
print('\nEvaluating the Gaussian Kernel ...\n')

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
sim = gaussian_kernel(x1, x2, sigma)

print('Gaussian kernel between x1 = [1, 2, 1], x2 = [0, 4, -1], sigma = {} : {:0.6f}\n'
      '(for sigma = 2, this value should be about 0.324652'.format(sigma, sim))

input('Program paused. Press enter to continue.\n')

# =============== Part 4: Visualizing Dataset 2 ================
print('Loading and Visualizing Data ...\n')

data = sio.loadmat('ex6data2.mat')
X, y = data['X'], data['y'].flatten()

plot_data(X, y)
plt.show()

input('Program paused. Press enter to continue.\n')

# ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
print('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n')

# SVM parameters
C = 1
sigma = 0.1
gamma = 1.0 / (2.0 * sigma ** 2)

model = svm.SVC(C, kernel='rbf', gamma=gamma)
model.fit(X, y)

visualize_boundary(X, y, model)
plt.show()

input('Program paused. Press enter to continue.\n')

# =============== Part 6: Visualizing Dataset 3 ================
print('Loading and Visualizing Data ...\n')

data = sio.loadmat('ex6data3.mat')
X, y = data['X'], data['y'].flatten()
Xval = data['Xval']
yval = data['yval'].flatten()

plot_data(X, y)
plt.show()

input('Program paused. Press enter to continue.\n')

# ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========
C, sigma = dataset_3_params(X, y, Xval, yval)

# Train the SVM
gamma = 1.0 / (2.0 * sigma ** 2)
model = svm.SVC(C, kernel='rbf', gamma=gamma)
model.fit(X, y)

visualize_boundary(X, y, model)
plt.show()

input('Program paused. Press enter to continue.\n')
