import numpy as np
import matplotlib.pyplot as plt
from plotData import plot_data


def visualize_boundary(X, y, model):
    plot_data(X, y)

    x1plot = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    x2plot = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)

    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(X1.shape)

    for i in range(X1.shape[1]):
        this_X = np.c_[X1[:, i], X2[:, i]]
        vals[:, i] = model.predict(this_X)

    plt.contour(X1, X2, vals, colors='b', levels=[0])
