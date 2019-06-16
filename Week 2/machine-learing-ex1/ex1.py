from matplotlib import cm
from matplotlib.colors import LogNorm

import gradientDescent
import warmUpExercise
import os
import pandas as pd
import numpy as np
import plotData
import matplotlib.pyplot as plt
import computeCost
from mpl_toolkits.mplot3d import Axes3D

# ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.py
print('Running warmUpExercise ... ')
print('5x5 Identity Matrix: ')
print(warmUpExercise.warm_up_exercise())
print('Program paused. Press enter to continue.')
os.system('pause')

# ======================= Part 2: Plotting =======================
print('Plotting Data ...')
data = np.loadtxt('ex1data1.txt', delimiter=',')
X = data[:, 0]
y = data[:, 1]
m = len(y)  # number of training examples

# plot data
# Note: You have to complete the code in plotData.py
plotData.plot_data(X, y)
print('Program paused. Press enter to continue.')
os.system('pause')

# =================== Part 3: Cost and Gradient descent ===================
X = np.c_[np.ones(m), X]  # Add a column of ones to x
theta = np.zeros(2)  # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

# compute and display initial cost
print('Testing the cost function ...')
J = computeCost.computer_cost(X, y, theta)
print('With theta = [0 ; 0]\nCost computed = {:.2f}'.format(J))
print('Expected cost value (approx) 32.07\n')

# further testing of the cost function
J = computeCost.computer_cost(X, y, np.array([-1, 2]))
print('With theta = [-1 ; 2]\nCost computed = {:.2f}'.format(J))
print('Expected cost value (approx) 54.24')

print('Program paused. Press enter to continue.')
os.system('pause')

print('Running Gradient Descent ...')
# run gradient descent
theta = gradientDescent.gradient_descent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent:')
print(theta)
print('Expected theta values (approx)')
print(' -3.6303\n  1.1664\n')

# Plot the linear fit
# plt.plot(X[:, 1], y, 'rx', label='Training data')
# plt.xlabel('Population of City in 10,000s')
# plt.ylabel('Profit in $10,000s')
plt.plot(X[:, 1], X @ theta, '-', label='Linear regression')
plt.legend(loc='lower right')
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.vdot(np.array([1, 3.5]).reshape(2, 1), theta)
print('For population = 35,000, we predict a profit of {}'.format(predict1 * 10000))
predict2 = np.vdot(np.array([1, 7]).reshape(2, 1), theta)
print('For population = 70,000, we predict a profit of {}'.format(predict2 * 10000))

print('Program paused. Press enter to continue.')
os.system('pause')

# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...')

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((theta0_vals.size, theta1_vals.size))
xs, ys = np.meshgrid(theta0_vals, theta1_vals)

# Fill out J_vals
for i in range(theta0_vals.size):
    for j in range(theta1_vals.size):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i, j] = computeCost.computer_cost(X, y, t)

# % Because of the way meshgrids work in the surf command, we need to
# % transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.T

# Surface plot
fig1 = plt.figure(1)
ax = fig1.gca(projection='3d')
ax.plot_surface(xs, ys, J_vals, cmap=cm.get_cmap('gist_rainbow_r'))
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.show()

# Contour plot
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100

fig2 = plt.figure(2)
plt.contour(xs, ys, J_vals, levels=np.logspace(-2, 3, 20), norm=LogNorm())
plt.plot(theta[0], theta[1], 'rx')
plt.show()

input('ex1 Finished. Press ENTER to exit')
