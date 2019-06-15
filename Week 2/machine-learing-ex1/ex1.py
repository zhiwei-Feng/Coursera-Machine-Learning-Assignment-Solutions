import warmUpExercise
import os
import pandas as pd
import numpy as np

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
