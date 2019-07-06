import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from plotData import plot_data
from linearKernel import linear_kernel
from sklearn import svm
from visualizeBoundaryLinear import visualize_boundary_linear

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
