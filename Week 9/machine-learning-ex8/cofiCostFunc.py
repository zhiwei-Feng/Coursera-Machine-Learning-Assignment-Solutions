import numpy as np


def cofi_cost_func(params, Y, R, num_users, num_movies, num_features, Lambda):
    # Unfold the U and W matrices from params
    X = np.reshape(params[:num_movies*num_features],
                   newshape=(num_movies, num_features))
    Theta = np.reshape(params[num_movies*num_features:],
                       newshape=(num_users, num_features))

    # You need to return the following values correctly
    J = 0
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    '''
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the cost function and gradient for collaborative
    %               filtering. Concretely, you should first implement the cost
    %               function (without regularization) and make sure it is
    %               matches our costs. After that, you should implement the 
    %               gradient and use the checkCostFunction routine to check
    %               that the gradient is correct. Finally, you should implement
    %               regularization.
    %
    % Notes: X - num_movies  x num_features matrix of movie features
    %        Theta - num_users  x num_features matrix of user features
    %        Y - num_movies x num_users matrix of user ratings of movies
    %        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
    %            i-th movie was rated by the j-th user
    %
    % You should set the following variables correctly:
    %
    %        X_grad - num_movies x num_features matrix, containing the 
    %                 partial derivatives w.r.t. to each element of X
    %        Theta_grad - num_users x num_features matrix, containing the 
    %                     partial derivatives w.r.t. to each element of Theta
    %
    '''
    hypothesis = (X @ Theta.T - Y) * R
    J = np.sum(hypothesis ** 2) / 2

    grad = np.r_[X_grad.flatten(), Theta_grad.flatten()]
    return J, grad
