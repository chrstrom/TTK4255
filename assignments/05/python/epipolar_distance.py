import numpy as np
from numba import jit

@jit(nopython=True) 
def epipolar_distance(F, uv1, uv2):
    """
    F should be the fundamental matrix (use F_from_E)
    uv1, uv2 should be 3 x n homogeneous pixel coordinates
    """
    n = uv1.shape[1]
    e = np.zeros(n)

    for i in range(n):
        u1 = uv1[:, i]
        u2 = uv2[:, i]
        F1 = F @ u1
        F2 = F.T @ u2

        e1 = u1.T @ F.T @ u2 / np.sqrt(F2[0] ** 2 + F2[1] ** 2)
        e2 = u2.T @ F @ u1 / np.sqrt(F1[0] ** 2 + F1[1] ** 2)

        e[i] = 0.5 * (e1 + e2)

    return e
