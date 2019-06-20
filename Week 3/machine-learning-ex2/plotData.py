import numpy as np
import matplotlib.pyplot as plt


def plot_data(X, y):
    # '+' for the positive example,
    # 'o' for the negative example
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    plt.figure(figsize=(11, 7))
    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='b', label='Admitted')
    plt.scatter(X[neg, 0], X[neg, 1], marker='o', c='y', label='Not admitted')

