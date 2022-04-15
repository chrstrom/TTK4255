import numpy as np
import matplotlib.pyplot as plt
from hw5.common import *
from scipy.optimize import least_squares
from match_features import FeatureMatcher
from matlab_inspired_interface import match_features



def draw_frame(ax, T, scale):
    X0 = T @ np.array((0, 0, 0, 1))
    X1 = T @ np.array((1, 0, 0, 1))
    X2 = T @ np.array((0, 1, 0, 1))
    X3 = T @ np.array((0, 0, 1, 1))
    ax.plot([X0[0], X1[0]], [X0[2], X1[2]], [X0[1], X1[1]], color="#FF7F0E")
    ax.plot([X0[0], X2[0]], [X0[2], X2[2]], [X0[1], X2[1]], color="#2CA02C")
    ax.plot([X0[0], X3[0]], [X0[2], X3[2]], [X0[1], X3[1]], color="#1F77B4")


def draw_point_cloud(X, T_m2q, xlim, ylim, zlim, colors, marker_size, frame_size):
    ax = plt.axes(projection="3d")
    ax.set_box_aspect((1, 1, 1))
    if colors.max() > 1.1:
        colors = colors.copy() / 255
    ax.scatter(
        X[0, :], X[2, :], X[1, :], c=colors, marker=".", s=marker_size, depthshade=False
    )
    draw_frame(ax, np.linalg.inv(T_m2q), scale=frame_size)
    ax.grid(False)
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_zlim([ylim[1], ylim[0]])
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    plt.title("[Click, hold and drag with the mouse to rotate the view]")

def T_from_h(h):
    p = h[:3]
    t = h[3:]
    R = rotate_x(p[0])@rotate_y(p[1])@rotate_z(p[2])@R0

    T = np.eye(4)
    T[:3, :3] = R[:3, :3]
    T[:3, 3] = t
    return T

def resfun(h):
    T = T_from_h(h)
    u_hat = project(K, T@X)
    N = u.shape[0]
    M = u.shape[1]
    r = np.zeros(N*M)
    for i in range(M):
        r[i] = u_hat[0, i] - u[0, i]
        r[i+M] = u_hat[1, i] - u[1, i]

    return r


# Tip: The solution from HW4 is inside common.py

# K = np.loadtxt('../data/K.txt')
# u = np.loadtxt('../data/platform_corners_image_minus_one.txt') #each column is (ui, vi).T 
# X = np.loadtxt('../data/platform_corners_metric_minus_one.txt') # each column is (X Y 0 1).T
# I = plt.imread('../data/img_sequence/video0000.jpg') # Only used for plotting

K = np.loadtxt("../data/calibration/K.txt")
XM = np.load("../localization/X.npy")
I = plt.imread("../data/undistorted/IMG_8210.jpg") 
model_desc = np.load("../localization/desc.npy")
options = [30000, 4, 0.001, 5, 1.5, 0.9, True]
feature_matcher = FeatureMatcher(options)


keypoints, desc = feature_matcher.get_keypoints(I)
index_pairs, match_metric = match_features(desc, model_desc, 0.9, True)
N = min(keypoints.shape[0], XM.shape[1], index_pairs.shape[0])

u = np.zeros((N, 2))
X = np.zeros((N, 4))
for i in range(N):
    u[i, :] = keypoints[index_pairs[i, 0]]
    X[i, :] = XM.T[index_pairs[i, 1]]


# Task 2.1 (from HW4)
u = u.T
X = X.T
uv1 = np.vstack((u, np.ones(u.shape[1])))

xy_tilde = np.linalg.inv(K) @ uv1
xy = xy_tilde / xy_tilde[2,:]
xy = xy_tilde[:2, :]
XY = X[:2, :]

H = estimate_H(xy, XY)

T1, T2 = decompose_H(H)
translation_z = T1[2, 3]
T_linear = T1 if translation_z < 0 else T2
R0 = np.eye(4)
R0[:3, :3] = T_linear[:3, :3]

x0 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
h = least_squares(resfun, x0, method='lm').x

T_hat = T_from_h(h).T

colors = np.zeros((X.shape[1], 3))

# These control the visible volume in the 3D point cloud plot.
# You may need to adjust these if your model does not show up.
xlim = [-10, +10]
ylim = [-10, +10]
zlim = [0, +20]

frame_size = 1
marker_size = 5

plt.figure("3D point cloud", figsize=(6, 6))
draw_point_cloud(
    X,
    T_hat,
    xlim,
    ylim,
    zlim,
    colors=colors,
    marker_size=marker_size,
    frame_size=frame_size,
)
plt.tight_layout()
plt.show()
