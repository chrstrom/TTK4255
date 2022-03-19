import numpy as np


def triangulate_many(xy1, xy2, P1, P2):
    """
    Arguments
        xy: Calibrated image coordinates in image 1 and 2
            [shape 3 x n]
        P:  Projection matrix for image 1 and 2
            [shape 3 x 4]
    Returns
        X:  Dehomogenized 3D points in world frame
            [shape 4 x n]
    """
    n = xy1.shape[1]
    X = np.empty((4, n))

    for i in range(n):
        # solve for (9)
        A = np.array([xy1[0, i] * P1[2,:] - P1[0,:], 
        xy1[1, i] * P1[2,:] - P1[1,:], 
        xy2[0, i] * P2[2,:] - P2[0,:], 
        xy2[1, i] * P2[2,:] - P2[1,:]])

        _, _, VT = np.linalg.svd(A)

        X[:, i] = VT[-1]

    return X / X[-1]
