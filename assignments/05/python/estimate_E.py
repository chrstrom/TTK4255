import numpy as np
import scipy.linalg as la


def estimate_E(xy1, xy2):
    n = xy1.shape[1]
    Q = np.empty((n, 9))

    for row in range(n):
        xl = xy1[0, row]
        yl = xy1[1, row]
        xr = xy2[0, row]
        yr = xy2[1, row]
        Q[row] = np.array([xr * xl, xr * yl, xr, yr * xl, yr * yl, yr, xl, yl, 1])

    _, _, VT = np.linalg.svd(Q)
    return VT[-1].reshape((3, 3))
