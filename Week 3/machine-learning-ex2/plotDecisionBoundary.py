import numpy as np
import matplotlib.pyplot as plt

from mapFeature import map_feature
from plotData import plot_data


def plot_decision_boundary(theta, X, y):
    plot_data(X[:, 1:3], y)

    if X.shape[1] <= 3:
        plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])
        plot_y = (-1 / theta[2] * (theta[1] * plot_x + theta[0]))

        plt.plot(plot_x, plot_y, label='Decision Boundary')

        plt.legend(loc='upper right')
        plt.axis([30, 100, 30, 100])
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros(len(u), len(v))

        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = map_feature(u[i], v[i]) @ theta

        z = z.T

        cs = plt.contour(u, v, z, levels=[0], colors='r', label='Decision Boundary')
        plt.legend([cs.collections[0]], ['Decision Boundary'])
