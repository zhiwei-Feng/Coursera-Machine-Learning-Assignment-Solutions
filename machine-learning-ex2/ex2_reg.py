import numpy as np
import matplotlib.pyplot as plt
from plotData import plot_data
from mapFeature import map_feature
from costFunctionReg import cost_function_reg
import scipy.optimize as op
from predict import predict
from plotDecisionBoundary import plot_decision_boundary

np.set_printoptions(precision=4, suppress=True)

data = np.loadtxt('ex2data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]

plot_data(X, y)

plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

plt.legend(('y = 1', 'y = 0'))
plt.show()

# =========== Part 1: Regularized Logistic Regression ============
# Add Polynomial Features
X = map_feature(X[:, 0], X[:, 1])

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

lamb = 1

cost, grad = cost_function_reg(initial_theta, X, y, lamb)
print('Cost at initial theta (zeros): {}\n'.format(cost))
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros) - first five values only:\n')
print(' {} \n'.format(grad[:5]))
print('Expected gradients (approx) - first five values only:\n')
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')

input('\nProgram paused. Press enter to continue.\n')

# Compute and display cost and gradient with all-ones theta and lambda = 10
test_theta = np.ones(X.shape[1])
cost, grad = cost_function_reg(test_theta, X, y, 10)

print('\nCost at test theta (with lambda = 10): {}\n'.format(cost))
print('Expected cost (approx): 3.16\n')
print('Gradient at test theta - first five values only:\n')
print(' {} \n'.format(grad[:5]))
print('Expected gradients (approx) - first five values only:\n')
print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n')

input('\nProgram paused. Press enter to continue.\n')

# ============= Part 2: Regularization and Accuracies =============
initial_theta = np.zeros(X.shape[1])

lamb = 1


def cost_f(t):
    return cost_function_reg(t, X, y, lamb)[0]


def gradient(t):
    return cost_function_reg(t, X, y, lamb)[1]


# Optimize
res = op.minimize(cost_f, x0=initial_theta, method='Newton-CG', jac=gradient)
theta, J, exit_flag = res.x, res.fun, res.success

plot_decision_boundary(theta, X, y)
plt.title('lambda = {}'.format(lamb))
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(('y = 1', 'y = 0', 'Decision boundary'))
plt.show()

# Compute accuracy on our training set


p = predict(theta, X)

print('Train Accuracy: {}\n'.format(np.mean(p == y) * 100))
print('Expected accuracy (with lambda = 1): 83.1 (approx)\n')
