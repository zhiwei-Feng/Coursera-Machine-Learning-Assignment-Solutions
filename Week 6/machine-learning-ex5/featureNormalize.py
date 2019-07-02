import numpy as np


def feature_normalize(X):
    """
    % ====================== YOUR CODE HERE ======================
    % Instructions: First, for each feature dimension, compute the mean
    %               of the feature and subtract it from the dataset,
    %               storing the mean value in mu. Next, compute the
    %               standard deviation of each feature and divide
    %               each feature by it's standard deviation, storing
    %               the standard deviation in sigma.
    %
    %               Note that X is a matrix where each column is a
    %               feature and each row is an example. You need
    %               to perform the normalization separately for
    %               each feature.
    %
    % Hint: You might find the 'mean' and 'std' functions useful.
    %
    """
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma
