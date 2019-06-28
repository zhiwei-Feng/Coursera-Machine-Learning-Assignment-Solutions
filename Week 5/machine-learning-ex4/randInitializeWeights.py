import numpy as np


def rand_init_weights(l_in, l_out):
    ep_init = 0.12
    w = np.random.rand(l_out, 1 + l_in) * (2 * ep_init) - ep_init
    return w
