import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from cofiCostFunc import cofi_cost_func
from checkCostFunction import check_cost_function

plt.ion()

# =============== Part 1: Loading movie ratings dataset ================
print('Loading movie ratings dataset.\n\n')

# Load data
data = loadmat('ex8_movies.mat')
Y, R = data['Y'], data['R']

'''
%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 
%  943 users
%
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i
'''

# From the matrix, we can compute statistics like average rating.
print('Average rating for movie 1 (Toy Story): {} / 5\n\n'
      .format(np.mean(Y[0, R[0, :]])))

# We can "visualize" the ratings matrix by plotting it with imagesc
plt.imshow(Y, extent=[0, 1, 0, 1])
plt.xlabel('Movies')
plt.ylabel('Users')

input('\nProgram paused. Press enter to continue.\n')

# ============ Part 2: Collaborative Filtering Cost Function ===========
data = loadmat('ex8_movieParams.mat')
X = data['X']
Theta = data['Theta']
num_users = data['num_users'].flatten()
num_movies = data['num_movies'].flatten()
num_features = data['num_features'].flatten()

# Reduce the data set size so that this runs faster
num_users = 4
num_movies = 5
num_features = 3
X = X[:num_movies, :num_features]
Theta = Theta[:num_users, :num_features]
Y = Y[:num_movies, :num_users]
R = R[:num_movies, :num_users]

# Evaluate cost function
J, _ = cofi_cost_func(np.r_[X.flatten(), Theta.flatten()],
                      Y, R, num_users, num_movies, num_features, 0)

print("Cost at loaded parameters: {} \n (this value should be about 22.22)"
      .format(J))

input('\nProgram paused. Press enter to continue.\n')

# ============== Part 3: Collaborative Filtering Gradient ==============
print('\nChecking Gradients (without regularization) ... ')

# Check gradients by running checkNNGradients
check_cost_function(0)

input('\nProgram paused. Press enter to continue.\n')

# ========= Part 4: Collaborative Filtering Cost Regularization ========
# Evaluate cost function
J, _ = cofi_cost_func(np.r_[X.flatten(), Theta.flatten()],
                      Y, R, num_users, num_movies, num_features, 1.5)

print('Cost at loaded parameters (lambda = 1.5): {} \n (this value should be \
      about 31.34)'.format(J))

input('\nProgram paused. Press enter to continue.\n')

# ======= Part 5: Collaborative Filtering Gradient Regularization ======
print('\nChecking Gradients (with regularization) ... ')

# Check gradients by running checkNNGradients
check_cost_function(1.5)

input('\nProgram paused. Press enter to continue.\n')
