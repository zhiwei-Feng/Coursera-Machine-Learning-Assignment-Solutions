import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from linearRegCostFunction import linear_reg_cost_function
from trainLinearReg import train_linear_reg
from learningCurve import learning_curve
from polyFeatures import poly_features
from featureNormalize import feature_normalize

# =========== Part 1: Loading and Visualizing Data =============
# Load Training Data
print('Loading and Visualizing Data ...\n')

# You will have X, y, Xval, yval, Xtest, ytest in your environment
data = sio.loadmat('ex5data1.mat')
X, y, Xval, yval, Xtest, ytest = data['X'], data['y'].flatten(), data['Xval'], data['yval'].flatten(), \
                                 data['Xtest'], data['ytest'].flatten()

# m = Number of examples
m = X.shape[0]

# Plot training data
plt.plot(X, y, 'rx', linewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

input('Program paused. Press enter to continue.\n')

# =========== Part 2: Regularized Linear Regression Cost =============
theta = np.array([1, 1])
J, _ = linear_reg_cost_function(np.c_[np.ones(m), X], y, theta, 1)

print('Cost at theta = [1 ; 1]: {} \n(this value should be about 303.993192)\n'.format(J))

input('Program paused. Press enter to continue.\n')

# =========== Part 3: Regularized Linear Regression Gradient =============
theta = np.array([1, 1])
J, grad = linear_reg_cost_function(np.c_[np.ones(m), X], y, theta, 1)

print('Gradient at theta = [1 ; 1]:  [{}; {}] \n(this value should be about [-15.303016; 598.250744])\n'
      .format(grad[0], grad[1]))

input('Program paused. Press enter to continue.\n')

# =========== Part 4: Train Linear Regression =============
lmd = 0
theta = train_linear_reg(np.c_[np.ones(m), X], y, lmd)

# Plot fit over the data
plt.plot(X, y, 'rx', linewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.plot(X, np.c_[np.ones(m), X] @ theta, '-', linewidth=2)
plt.show()

input('Program paused. Press enter to continue.\n')

# =========== Part 5: Learning Curve for Linear Regression =============
lmd = 0
error_train, error_val = learning_curve(np.c_[np.ones(m), X], y, np.c_[np.ones(Xval.shape[0]), Xval], yval, lmd)

l1, l2 = plt.plot(np.arange(1, m + 1), error_train, np.arange(1, m + 1), error_val)
plt.title('Learning curve for linear regression')
plt.legend((l1, l2), ('Train', 'Cross Validation'))
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 150])
plt.show()

print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(m):
    print('  \t{}\t\t{}\t{}\n'.format(i, error_train[i], error_val[i]))

input('Program paused. Press enter to continue.\n')

# =========== Part 6: Feature Mapping for Polynomial Regression =============
p = 8

# Map X onto Polynomial Features and Normalize
X_poly = poly_features(X, p)
X_poly, mu, sigma = feature_normalize(X_poly)
X_poly = np.c_[np.ones(m), X_poly]

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = poly_features(Xtest, p)
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test / sigma
X_poly_test = np.c_[np.ones(X_poly_test.shape[0]), X_poly_test]

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = poly_features(Xval, p)
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val / sigma
X_poly_val = np.c_[np.ones(X_poly_val.shape[0]), X_poly_val]

print('Normalized Training Example 1:\n')
print('  {}  \n'.format(X_poly[0]))

input('Program paused. Press enter to continue.\n')

# =========== Part 7: Learning Curve for Polynomial Regression =============