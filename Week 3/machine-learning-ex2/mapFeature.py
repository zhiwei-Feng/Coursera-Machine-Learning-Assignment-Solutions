def map_feature(x1, x2):
    degree = 6
    import numpy as np
    out = np.ones(x1.shape[0])
    for i in range(1, degree + 1):
        for j in range(i):
            out = np.c_(out, (x1 ** (i - j) * (x2 ** j)))
    return out
