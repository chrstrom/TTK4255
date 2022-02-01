#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt


def translate_xyz(x = 0, y = 0, z = 0):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])

def rotate_x(rad):
    c = np.cos(rad)
    s = np.sin(rad)
    return np.array([[1, 0,  0, 0],
                     [0, c, -s, 0],
                     [0, s,  c, 0],
                     [0, 0,  0, 1]])

def rotate_y(rad):
    c = np.cos(rad)
    s = np.sin(rad)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]])

def rotate_z(rad):
    c = np.cos(rad)
    s = np.sin(rad)
    return np.array([[c, -s, 0, 0],
                     [s,  c, 0, 0],
                     [0,  0, 1, 0],
                     [0,  0, 0, 1]])


def project(K, X):
    """
    Computes the pinhole projection of a 3xN array of 3D points X
    using the camera intrinsic matrix K. Returns the dehomogenized
    pixel coordinates as an array of size 2xN.
    """
    if X.shape[0] == 3:
        homogenous_coordinate_vector = K @ X
    elif X.shape[0] == 4:
        homogenous_coordinate_vector = K @ X[:3, :]

    u_tilde = homogenous_coordinate_vector[0, :]
    v_tilde = homogenous_coordinate_vector[1, :]
    w_tilde = homogenous_coordinate_vector[2, :]

    u = u_tilde / w_tilde
    v = v_tilde / w_tilde

    return u, v


def draw_frame(K, T, scale=1):
    """
    Visualize the coordinate frame axes of the 4x4 object-to-camera
    matrix T using the 3x3 intrinsic matrix K.
    This uses your project function, so implement it first.
    Control the length of the axes using 'scale'.
    """
    X = T @ np.array([
        [0,scale,0,0],
        [0,0,scale,0],
        [0,0,0,scale],
        [1,1,1,1]])
    u,v = project(K, X) # If you get an error message here, you should modify your project function to accept 4xN arrays of homogeneous vectors, instead of 3xN.
    plt.plot([u[0], u[1]], [v[0], v[1]], color='red') # X-axis
    plt.plot([u[0], u[2]], [v[0], v[2]], color='green') # Y-axis
    plt.plot([u[0], u[3]], [v[0], v[3]], color='blue') # Z-axis