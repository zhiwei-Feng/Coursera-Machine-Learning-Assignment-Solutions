import matplotlib.pyplot as plt
import numpy as np


def plot_data(X, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)

    plt.plot(X[pos, 0], X[pos, 1], 'k+', markersize=7, linewidth=1)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', markersize=7, markerfacecolor='y')
