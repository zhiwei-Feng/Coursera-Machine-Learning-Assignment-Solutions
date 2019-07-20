import numpy as np
import matplotlib.pyplot as plt

from costFunction import cost_function
from plotData import plot_data
import scipy.optimize as op

# Load Data
from plotDecisionBoundary import plot_decision_boundary
from predict import predict
from sigmoid import sigmoid

data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]

# ==================== Part 1: Plotting ====================
print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')
plot_data(X, y)
plt.legend(loc='upper right')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.show()

input('\nProgram paused. Press enter to continue.\n')

# ============ Part 2: Compute Cost and Gradient ============
m, n = X.shape

# Add intercept term to X
X = np.c_[np.ones(m), X]

# initial fitting parameters
initial_theta = np.zeros(n + 1)
cost, grad = cost_function(initial_theta, X, y)

print('Cost at initial theta (zeros): {}\n'.format(cost))
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros): \n')
print('{} \n'.format(grad))
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2])
cost, grad = cost_function(test_theta, X, y)

print('\nCost at test theta: {}\n'.format(cost))
print('Expected cost (approx): 0.218\n')
print('Gradient at test theta: \n')
print('{} \n'.format(grad))
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

input('\nProgram paused. Press enter to continue.\n')


# ============= Part 3: Optimizing using fminunc  =============
def cost_f(t):
    return cost_function(t, X, y)[0]


def gradient(t):
    return cost_function(t, X, y)[1]


# cost_f() return only one value,gradient() return only one value
res = op.minimize(cost_f, x0=initial_theta, method='Newton-CG', jac=gradient)
theta, cost = (res.x, res.fun)

# Print theta to screen
print('Cost at theta found by fminunc: {}\n'.format(cost))
print('Expected cost (approx): 0.203\n')
print('theta: \n')
print(' {} \n'.format(theta))
print('Expected theta (approx):\n')
print(' -25.161\n 0.206\n 0.201\n')

# Plot Boundary
plot_decision_boundary(theta, X, y)

plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.show()

input('\nProgram paused. Press enter to continue.\n')

# ============== Part 4: Predict and Accuracies ==============
prob = sigmoid(np.vdot(np.array([1, 45, 85]), theta))
print('For a student with scores 45 and 85, we predict an admission probability of {}\n'.format(prob))
print('Expected value: 0.775 +/- 0.002\n\n')

p = predict(theta, X)
print('Train Accuracy: {}\n'.format(np.mean(p == y) * 100))
print('Expected accuracy (approx): 89.0\n')
