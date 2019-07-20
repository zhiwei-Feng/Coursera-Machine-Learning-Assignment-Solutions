import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from displayData import display_data
from nnCostFunction import nn_cost_function
from sigmoid import sigmoid_gradient
from randInitializeWeights import rand_init_weights
from checkNNGradients import checkNNGradients
from scipy.optimize import minimize, fmin_cg
from predict import predict

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

# =========== Part 1: Loading and Visualizing Data =============
# Load Data
print('Loading and Visualizing Data ...\n')

data = sio.loadmat('ex4data1.mat')
X, y = data['X'], data['y'].flatten() # flatten is important!
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

J, grad = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambd)

print('Cost at parameters (loaded from ex4weights): {} \n(this value should be about 0.287629)\n'.format(J))

input('Program paused. Press enter to continue.\n')

# =============== Part 4: Implement Regularization ===============
print('\nChecking Cost Function (w/ Regularization) ... \n')

lambd = 1

J, grad = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambd)

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

initial_nn_params = np.r_[initial_Theta1.flatten(), initial_Theta2.flatten()]

# =============== Part 7: Implement Backpropagation ===============
print('\nChecking Backpropagation... \n')

checkNNGradients()

input('\nProgram paused. Press enter to continue.\n')

# =============== Part 8: Implement Regularization ===============
print('\nChecking Backpropagation (w/ Regularization) ... \n')

Lambda = 3
checkNNGradients(Lambda)

debug_J, _ = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)

print('\n\nCost at (fixed) debugging parameters (w/ lambda = {}): {} \n(for lambda = 3, this value should be about '
      '0.576051)\n\n'.format(Lambda, debug_J))

input('\nProgram paused. Press enter to continue.\n')

# =================== Part 9: Training NN ===================
print('\nTraining Neural Network... \n')

Lambda = 0.5


def cost_func(t):
    return nn_cost_function(t, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)[0]


def grad_func(t):
    return nn_cost_function(t, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)[1]


res = minimize(cost_func, x0=initial_nn_params, method='CG', jac=grad_func, options={'maxiter': 50, 'disp': True})
# nn_params, cost = fmin_cg(cost_func, fprime=grad_func, x0=initial_nn_params, maxiter=50, disp=True, full_output=True)
nn_params, cost = res.x, res.fun
# Obtain Theta1 and Theta2 back from nn_params

Theta1 = np.reshape(nn_params[0:(hidden_layer_size * (input_layer_size + 1))],
                    (hidden_layer_size, (input_layer_size + 1)))  # (25,401)
Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                    (num_labels, (hidden_layer_size + 1)))  # (10,26)

input('\nProgram paused. Press enter to continue.\n')

# ================= Part 9: Visualize Weights =================
print('\nVisualizing Neural Network... \n')

display_data(Theta1[:, 1:])
plt.show()

input('\nProgram paused. Press enter to continue.\n')

# ================= Part 10: Implement Predict =================
pred = predict(Theta1, Theta2, X)
print('Training Set Accuracy: {:.1f}%'.format(np.mean(pred == y) * 100))
