import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from cofiCostFunc import cofi_cost_func
from checkCostFunction import check_cost_function
from loadMovieList import load_movie_list
from normalizeRatings import normalize_ratings
from scipy.optimize import minimize, fmin_cg

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

# ============== Part 6: Entering ratings for a new user ===============
movie_list = load_movie_list()

# Initialize my ratings
my_ratings = np.zeros(1682)

my_ratings[0] = 4
my_ratings[97] = 2

my_ratings[6] = 3
my_ratings[11] = 5
my_ratings[53] = 4
my_ratings[63] = 5
my_ratings[65] = 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5

print('\n\nNew user ratings:')
for i in range(my_ratings.size):
    if my_ratings[i] > 0:
        print('Rated {} for {}'.format(my_ratings[i], movie_list[i]))

input('\nProgram paused. Press enter to continue.\n')

# ================== Part 7: Learning Movie Ratings ====================
print('Training collaborative filtering ...\n'
      '(this may take 1 ~ 2 minutes)')


# Load data
data = loadmat('ex8_movies.mat')
Y = data['Y']
R = data['R']

# Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by
# 943 users
#
# R is a 1682x943 matrix, where R[i,j] = 1 if and only if user j gave a
# rating to movie i

# Add our own ratings to the data matrix
Y = np.c_[my_ratings, Y]
R = np.c_[(my_ratings != 0), R]

# Normalize Ratings
Ynorm, Ymean = normalize_ratings(Y, R)

# Useful values
num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10

# Set initial parameters (theta, X)
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)

initial_params = np.r_[X.flatten(), Theta.flatten()]

lmd = 10


def cost_func(p):
    return cofi_cost_func(p, Ynorm, R, num_users, num_movies, num_features, lmd)[0]


def grad_func(p):
    return cofi_cost_func(p, Ynorm, R, num_users, num_movies, num_features, lmd)[1]


theta, *unused = fmin_cg(cost_func, x0=initial_params, fprime=grad_func,
                         maxiter=100, full_output=True)

# Unfold the returned theta back into U and W
X = theta[:num_movies * num_features].reshape((num_movies, num_features))
Theta = theta[num_movies * num_features:].reshape((num_users, num_features))

print('Recommender system learning completed')

input('Program paused. Press ENTER to continue')

# ===================== Part 8: Recommendation for you =====================
# After training the model, you can now make recommendations by computing
# the predictions matrix.
#

p = X @ Theta.T
my_predictions = p[:, 0] + Ymean

movieList = load_movie_list()

# sort predictions descending
pred_idxs_sorted = np.argsort(my_predictions)
pred_idxs_sorted[:] = pred_idxs_sorted[::-1]

print('\nTop recommendations for you:')
for i in range(10):
    j = pred_idxs_sorted[i]
    print('Predicting rating {:0.1f} for movie {}'.format(
        my_predictions[j], movie_list[j]))

print('\nOriginal ratings provided:')
for i in range(my_ratings.size):
    if my_ratings[i] > 0:
        print('Rated {} for {}'.format(my_ratings[i], movie_list[i]))

input('ex8_cofi Finished. Press ENTER to exit')
