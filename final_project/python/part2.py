import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from hw5.common import *
from hw5.estimate_E_ransac import *
from hw5.decompose_E import *
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
    N = uv1.shape[0]
    M = uv1.shape[1]
    r = np.zeros(N*M)
    for i in range(M):
        r[i] = u_hat[0, i] - uv1[0, i]
        r[i+M] = u_hat[1, i] - uv1[1, i]

    return r


K = np.loadtxt("../data/calibration/K.txt")
K_inv = la.inv(K)

XM = np.loadtxt("../localization/X_ivan.out")
model_desc = np.loadtxt("../localization/desc_ivan.out").astype("float32")
I = plt.imread("../data/undistorted/IMG_8214.jpg") 

options = [30000, 4, 0.001, 5, 1.5, 0.9, True]
feature_matcher = FeatureMatcher(options)

keypoints, desc = feature_matcher.get_keypoints(I)
index_pairs, match_metric = match_features(desc, model_desc, 0.9, True)
N = min(keypoints.shape[0], XM.shape[1], index_pairs.shape[0])

u = keypoints[index_pairs[:N, 0]].T
X = XM.T[index_pairs[:N, 1]].T

uv1 = np.vstack((u, np.ones(u.shape[1])))
uvw = K @ X[:3, :]
uv2 = uvw / uvw[2, :]

xy_tilde = K_inv @ uv1
xy = xy_tilde / xy_tilde[2,:]
xy = xy_tilde[:2, :]
XY = X[:2, :]

H = estimate_H(xy, XY)

T1, T2 = decompose_H(H)
translation_z = T1[2, 3]
T_linear = T1 if translation_z > 0 else T2
R0 = np.eye(4)
R0[:3, :3] = T_linear[:3, :3]

x0 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
h = least_squares(resfun, x0, method='lm').x

T_hat = T_from_h(h)

T_mq = np.eye(4)
T_mq[:3, :3] = T_hat[:3, :3].T
T_mq[:3, -1] = T_hat[:3, -1]
print(T_mq)

## RANSAC
xy1 = project(K_inv, uv1)
xy2 = project(K_inv, uv2)

distance_threshold = 4
inlier_fraction = 0.5
success_probability = 0.99
num_trials = np.ceil(np.log(1 - success_probability) / np.log(1 - inlier_fraction ** 8))
inlier_set = estimate_E_ransac(xy1, xy2, uv1, uv2, K, distance_threshold, int(num_trials))

uv1_in = uv1[:, inlier_set]
xy1_in = xy1[:, inlier_set]
xy2_in = xy2[:, inlier_set]


E = estimate_E(xy1_in, xy2_in).reshape((3, 3))
TS = decompose_E(E)

T_qm = TS[2]

T_mq = np.eye(4)
T_mq[:3, :3] = T_qm[:3, :3].T
T_mq[:3, -1] = -T_qm[:3, -1]

colors = I[uv1[1,:].astype(np.int32), uv1[0,:].astype(np.int32), :]

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
    T_mq,
    xlim,
    ylim,
    zlim,
    colors=colors,
    marker_size=marker_size,
    frame_size=frame_size,
)
plt.tight_layout()
plt.show()
