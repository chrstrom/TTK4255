import matplotlib.pyplot as plt
import numpy as np
from figures import *
from estimate_E import *
from decompose_E import *
from triangulate_many import *
from epipolar_distance import *
from F_from_E import *
from estimate_E_ransac import *


def project(K, X):
    X_tilde = K @ X
    return X_tilde[:2, :] / X_tilde[2, :]


K = np.loadtxt("../data/K.txt")
I1 = plt.imread("../data/image1.jpg") / 255.0
I2 = plt.imread("../data/image2.jpg") / 255.0
matches = np.loadtxt("../data/matches.txt")

uv1 = np.vstack([matches[:, :2].T, np.ones(matches.shape[0])])
uv2 = np.vstack([matches[:, 2:4].T, np.ones(matches.shape[0])])

K_inv = np.linalg.inv(K)

xy1 = project(K_inv, uv1)
xy2 = project(K_inv, uv2)

# # Task 2: Estimate E
E = estimate_E(xy1, xy2)



# F = F_from_E(E, K)

# # Task 3: Triangulate 3D points
# P1 = np.c_[np.eye(3), np.zeros(3)]
# P2 = decompose_E(E)[0]

# X = triangulate_many(xy1, xy2, P1, P2)

#
# Uncomment in Task 2
#
#np.random.seed(123)  # Leave as commented out to get a random selection each time
#draw_correspondences(I1, I2, uv1, uv2, F, sample_size=8)

#
# Uncomment in Task 3
# draw_point_cloud(X, I1, uv1, xlim=[-1, +1], ylim=[-1, +1], zlim=[1, 3])

# Task 4
matches = np.loadtxt('../data/task4matches.txt')
uv1 = np.vstack([matches[:, :2].T, np.ones(matches.shape[0])])
uv2 = np.vstack([matches[:, 2:4].T, np.ones(matches.shape[0])])

# e = epipolar_distance(F, uv1, uv2)
# n, bins, patches = plt.hist(e, 100, density=False, facecolor='k')
# plt.title("Average residuals (100 bins)")
# plt.xlabel("e")
# plt.ylabel("n")

xy1 = project(K_inv, uv1)
xy2 = project(K_inv, uv2)

inlier_fraction = 0.5 
success_probability = 0.99
num_trials = np.ceil(np.log(1 - success_probability) / np.log(1 - inlier_fraction**8))

E, inlier_set = estimate_E_ransac(xy1, xy2, uv1, uv2, K, 4, int(num_trials))

inliers = np.where(inlier_set == True)[0]
inlier_count = len(inliers)
uv1_in = np.zeros((3, inlier_count))
uv2_in = np.zeros((3, inlier_count))
xy1_in = np.zeros((2, inlier_count))
xy2_in = np.zeros((2, inlier_count))
for i, inlier in enumerate(inliers):
    uv1_in[:, i] = uv1[:, inlier]
    uv2_in[:, i] = uv2[:, inlier]
    xy1_in[:, i] = xy1[:, inlier]
    xy2_in[:, i] = xy2[:, inlier]

E = estimate_E(xy1_in, xy2_in)
F = F_from_E(E, K)

#sample_size = 8 if inlier_count > 8 else inlier_count 
sample_size = inlier_count
P1 = np.c_[np.eye(3), np.zeros(3)]
# Select P2 for the view that is in front of both cameras

max_points_in_front = 0
i_best = 0

# Iterate over all of the possible matrices and find the matrix with most
# measurements in front of the camera
P2s = decompose_E(E)
for i, P in enumerate(P2s):
    X = triangulate_many(xy1_in, xy2_in, P1, P)

    points_in_front = np.sum((P1 @ X)[2] > 0)
    if points_in_front > max_points_in_front:
        max_points_in_front = points_in_front
        i_best = i

P2 = P2s[i_best]

X = triangulate_many(xy1_in, xy2_in, P1, P2)
draw_correspondences(I1, I2, uv1_in, uv2_in, F, sample_size=sample_size)
draw_point_cloud(X, I1, uv1_in, xlim=[-1, +1], ylim=[-1, +1], zlim=[1, 3])
plt.show()
