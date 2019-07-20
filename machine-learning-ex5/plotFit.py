import numpy as np
import matplotlib.pyplot as plt
from polyFeatures import poly_features


def plot_fit(min_x, max_x, mu, sigma, theta, p):
    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x = np.arange(min_x - 15, max_x + 25, 0.05).T

    # Map the X values
    X_poly = poly_features(x, p)
    X_poly = X_poly - mu
    X_poly = X_poly / sigma

    # Add ones
    X_poly = np.column_stack((np.ones(x.shape[0]), X_poly))

    # Plot
    plt.plot(x, X_poly.dot(theta), '--', lw=2)
