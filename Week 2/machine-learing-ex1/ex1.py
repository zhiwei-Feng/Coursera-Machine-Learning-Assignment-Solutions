import warmUpExercise
import os
import pandas as pd
import numpy as np
import plotData
import matplotlib.pyplot as plt
import computeCost

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
X = np.column_stack((np.ones(m), X))  # Add a column of ones to x
theta = np.zeros((2, 1))  # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

print('Testing the cost function ...')
J = computeCost.computer_cost(X, y, theta)
print('With theta = [0 ; 0]\nCost computed = {:.2f}'.format(J))
print('Expected cost value (approx) 32.07\n')
