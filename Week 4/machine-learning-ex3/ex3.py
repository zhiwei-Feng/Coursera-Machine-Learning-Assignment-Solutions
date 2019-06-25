import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from displayData import display_data
from oneVsAll import one_vs_all
from predictOneVsAll import predict_one_vs_all
import lrCostFunction

np.set_printoptions(precision=6, suppress=True)

input_layer_size = 400
num_labels = 10

# Loading Training Data
print('Loading and Visualizing Data ...\n')

data = sio.loadmat('ex3data1.mat')
X, y = data['X'], data['y'].flatten()
m = y.size

# randomly select 100 data points to display
rand_indices = np.random.permutation(range(m))
sel = X[rand_indices[:100], :]

display_data(sel)
plt.show()

input('Program paused. Press enter to continue.\n')

# ============ Part 2a: Vectorize Logistic Regression ============
print('\nTesting lrCostFunction() with regularization')

theta_t = np.array([-2, -1, 1, 2])
X_t = np.c_[np.ones(5), np.arange(1, 16).reshape(3, 5).T / 10]
y_t = np.array([1, 0, 1, 0, 1])
lambda_t = 3
J = lrCostFunction.lrCostFunc(theta_t, X_t, y_t, lambda_t)
grad = lrCostFunction.lrGradFunc(theta_t, X_t, y_t, lambda_t)

print('\nCost: {}\n'.format(J))
print('Expected cost: 2.534819\n')
print('Gradients:\n')
print(' {} \n'.format(grad))
print('Expected gradients:\n')
print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n')

input('Program paused. Press enter to continue.\n')

# ============ Part 2b: One-vs-All Training ============
print('\nTraining One-vs-All Logistic Regression...\n')

lamb = 0.1

all_theta = one_vs_all(X, y, num_labels, lamb)
input('Program paused. Press enter to continue.\n')

# ================ Part 3: Predict for One-Vs-All ================
pred = predict_one_vs_all(all_theta, X)
print('Training Set Accuracy: {:.1f}'.format(np.mean(pred == y) * 100))
