import numpy as np


def select_threshold(yval, pval):
    bestEpsilon = 0
    bestF1 = 0

    stepsize = (max(pval)-min(pval)) / 1000
    r = np.arange(min(pval), max(pval), stepsize)
    for epsilon in r:
        predictions = np.less(pval, epsilon)
        tp = np.sum(np.logical_and(predictions, yval))
        fp = np.sum(np.logical_and(predictions, yval == 0))
        fn = np.sum(np.logical_and(np.logical_not(predictions), yval == 1))

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        F1 = (2 * precision * recall) / (precision + recall)

        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon

    return bestEpsilon, bestF1
