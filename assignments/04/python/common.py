#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def estimate_H(xy, XY):
    # Tip: U,s,VT = np.linalg.svd(A) computes the SVD of A.
    # The column of V corresponding to the smallest singular value
    # is the last column, as the singular values are automatically
    # ordered by decreasing magnitude. However, note that it returns
    # V transposed.

    n = XY.shape[0]
    A = np.zeros((2*n, 9))
    for i in range(n):
        Xi = XY[i, 0]
        Yi = XY[i, 1]
        xi = xy[i, 0]
        yi = xy[i, 1]
        A[2*i, :] = [Xi, Yi, 1, 0, 0, 0, -Xi*xi, -Yi*xi, -xi]
        A[2*i + 1, :] = [0, 0, 0, Xi, Yi, 1, -Xi*yi, -Yi*yi, -yi]

    _, _, VT = np.linalg.svd(A)
    V = VT.T

    h = V[:,-1]
    H = np.reshape(h, (3, 3))
    return H


def decompose_H(H, use_closest=False):
    # Tip: Use np.linalg.norm to compute the Euclidean length

    def calculate_T(H, k):
        r1 = H[:, 0] / k
        r2 = H[:, 1] / k
        r3 = np.cross(r1, r2)
        t = H[:, 2] / k

        T = np.eye(4)
        T[:3, :4] = np.column_stack((r1, r2, r3, t))

        if use_closest:
            T[:3, :3] = closest_rotation_matrix(np.column_stack((r1, r2, r3)))
        return T

    abs_k = np.linalg.norm(H[:, 0])
    T1 = calculate_T(H, abs_k)
    T2 = calculate_T(H, -abs_k)

    return T1, T2


def closest_rotation_matrix(Q):
    U, S, VT = np.linalg.svd(Q)
    R = U@VT
    return R


def project(K, X):
    """
    Computes the pinhole projection of an (3 or 4)xN array X using
    the camera intrinsic matrix K. Returns the dehomogenized pixel
    coordinates as an array of size 2xN.
    """
    uvw = K @ X[:3, :]
    uvw /= uvw[2, :]
    return uvw[:2, :]


def draw_frame(K, T, scale=1):
    """
    Visualize the coordinate frame axes of the 4x4 object-to-camera
    matrix T using the 3x3 intrinsic matrix K.

    Control the length of the axes by specifying the scale argument.
    """
    X = T @ np.array(
        [[0, scale, 0, 0], [0, 0, scale, 0], [0, 0, 0, scale], [1, 1, 1, 1]]
    )
    u, v = project(K, X)
    plt.plot([u[0], u[1]], [v[0], v[1]], color="red")  # X-axis
    plt.plot([u[0], u[2]], [v[0], v[2]], color="green")  # Y-axis
    plt.plot([u[0], u[3]], [v[0], v[3]], color="blue")  # Z-axis


def generate_figure(fig, image_number, K, T, uv, uv_predicted, XY):

    fig.suptitle("Image number %d" % image_number)

    #
    # Visualize reprojected markers and estimated object coordinate frame
    #
    I = plt.imread("../data/image%04d.jpg" % image_number)
    plt.subplot(121)
    plt.imshow(I)
    draw_frame(K, T, scale=4.5)
    plt.scatter(uv[0, :], uv[1, :], color="red", label="Detected")
    plt.scatter(
        uv_predicted[0, :],
        uv_predicted[1, :],
        marker="+",
        color="yellow",
        label="Predicted",
    )
    plt.legend()
    plt.xlim([0, I.shape[1]])
    plt.ylim([I.shape[0], 0])

    #
    # Visualize scene in 3D
    #
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.plot(XY[0, :], XY[1, :], np.zeros(XY.shape[1]), ".")  # Draw markers in 3D
    pO = np.linalg.inv(T) @ np.array([0, 0, 0, 1])  # Compute camera origin
    pX = np.linalg.inv(T) @ np.array([6, 0, 0, 1])  # Compute camera X-axis
    pY = np.linalg.inv(T) @ np.array([0, 6, 0, 1])  # Compute camera Y-axis
    pZ = np.linalg.inv(T) @ np.array([0, 0, 6, 1])  # Compute camera Z-axis
    plt.plot(
        [pO[0], pZ[0]], [pO[1], pZ[1]], [pO[2], pZ[2]], color="blue"
    )  # Draw camera Z-axis
    plt.plot(
        [pO[0], pY[0]], [pO[1], pY[1]], [pO[2], pY[2]], color="green"
    )  # Draw camera Y-axis
    plt.plot(
        [pO[0], pX[0]], [pO[1], pX[1]], [pO[2], pX[2]], color="red"
    )  # Draw camera X-axis
    ax.set_xlim([-40, 40])
    ax.set_ylim([-40, 40])
    ax.set_zlim([-25, 25])
    ax.set_xlabel("X")
    ax.set_zlabel("Y")
    ax.set_ylabel("Z")

    plt.tight_layout()
