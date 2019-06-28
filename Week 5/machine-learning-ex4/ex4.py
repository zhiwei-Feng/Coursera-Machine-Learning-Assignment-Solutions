import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from displayData import display_data
from nnCostFunction import nn_cost_function
from sigmoid import sigmoid_gradient
from randInitializeWeights import rand_init_weights

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

# =========== Part 1: Loading and Visualizing Data =============
# Load Data
print('Loading and Visualizing Data ...\n')

data = sio.loadmat('ex4data1.mat')
X, y = data['X'], data['y']
m = X.shape[0]

# Randomly select 100 data points to display
sel = np.random.permutation(range(m))
sel = sel[:100]

display_data(X[sel, :])
plt.show()

input('Program paused. Press enter to continue.\n')

# ================ Part 2: Loading Parameters ================
print('\nLoading Saved Neural Network Parameters ...\n')

# Load the weights into variables Theta1 and Theta2
weights = sio.loadmat('ex4weights.mat')
Theta1, Theta2 = weights['Theta1'], weights['Theta2']

# Unroll parameters
nn_params = np.r_[Theta1.flatten(), Theta2.flatten()]

# ================ Part 3: Compute Cost (Feedforward) ================
print('\nFeedforward Using Neural Network ...\n')

lambd = 0

J = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambd)

print('Cost at parameters (loaded from ex4weights): {} \n(this value should be about 0.287629)\n'.format(J))

input('Program paused. Press enter to continue.\n')

# =============== Part 4: Implement Regularization ===============
print('\nChecking Cost Function (w/ Regularization) ... \n')

lambd = 1

J = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambd)

print('Cost at parameters (loaded from ex4weights): {} \n(this value should be about 0.383770)\n'.format(J))

input('Program paused. Press enter to continue.\n')

# ================ Part 5: Sigmoid Gradient  ================
print('\nEvaluating sigmoid gradient...\n')

g = sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ')
print('{} '.format(g))
print()

input('Program paused. Press enter to continue.\n')

# ================ Part 6: Initializing Pameters ================
print('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = rand_init_weights(input_layer_size, hidden_layer_size)
initial_Theta2 = rand_init_weights(hidden_layer_size, num_labels)

initial_nn_params = np.r_[initial_Theta1.flatten(),initial_Theta2.flatten()]

# =============== Part 7: Implement Backpropagation ===============
print('\nChecking Backpropagation... \n')
