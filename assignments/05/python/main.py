import matplotlib.pyplot as plt
import numpy as np
from figures import *
from estimate_E import *
from decompose_E import *
from triangulate_many import *
from epipolar_distance import *
from estimate_E_ransac import *
from datetime import datetime

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


# F = K_inv.T @ E @ K_inv

# # Task 3: Triangulate 3D points
# P1 = np.c_[np.eye(3), np.zeros(3)]
# P2 = decompose_E(E)[0]

# X = triangulate_many(xy1, xy2, P1, P2)

#
# Uncomment in Task 2
#
# np.random.seed(123)  # Leave as commented out to get a random selection each time
# draw_correspondences(I1, I2, uv1, uv2, F, sample_size=8)

#
# Uncomment in Task 3
# draw_point_cloud(X, I1, uv1, xlim=[-1, +1], ylim=[-1, +1], zlim=[1, 3])

# Task 4
matches = np.loadtxt("../data/task4matches.txt")
uv1 = np.vstack([matches[:, :2].T, np.ones(matches.shape[0])])
uv2 = np.vstack([matches[:, 2:4].T, np.ones(matches.shape[0])])

# e = epipolar_distance(F, uv1, uv2)
# n, bins, patches = plt.hist(e, 100, density=False, facecolor='k')
# plt.title("Average residuals (100 bins)")
# plt.xlabel("e")
# plt.ylabel("n")

xy1 = project(K_inv, uv1)
xy2 = project(K_inv, uv2)

distance_threshold = 4
inlier_fraction = 0.5
success_probability = 0.99
num_trials = np.ceil(np.log(1 - success_probability) / np.log(1 - inlier_fraction ** 8))

time_start = datetime.now()
inlier_set = estimate_E_ransac(xy1, xy2, uv1, uv2, K, distance_threshold, int(num_trials))
print(f"Time taken: {datetime.now() - time_start}")

# total_inliers = 0
# num_ensemble = 100
# for i in range(num_ensemble):
#     time_start = datetime.now()
#     inlier_set = estimate_E_ransac(xy1, xy2, uv1, uv2, K, distance_threshold, int(num_trials))
#     total_inliers = total_inliers + len(inlier_set)
#     print(f"Iteration {i + 1} of {num_ensemble} ({datetime.now() - time_start})")
# # print(f"Time taken: {datetime.now() - time_start}")

# print(f"{num_ensemble}The  runs yielded an average inlier set size of {total_inliers / num_ensemble}")

uv1_in = uv1[:, inlier_set]
xy1_in = xy1[:, inlier_set]
xy2_in = xy2[:, inlier_set]

E = estimate_E(xy1_in, xy2_in).reshape((3, 3))

P1 = np.c_[np.eye(3), np.zeros(3)]

# Select P2 for the view that is in front of both cameras
max_points_in_front = 0
i_best = 0
P2s = decompose_E(E)
for i, P in enumerate(P2s):
    X = triangulate_many(xy1_in, xy2_in, P1, P)

    points_in_front = np.sum(X[2, :] > 0)
    if points_in_front > max_points_in_front:
        max_points_in_front = points_in_front
        i_best = i

P2 = P2s[i_best]

X = triangulate_many(xy1_in, xy2_in, P1, P2)
draw_point_cloud(X, I1, uv1_in, xlim=[-1, +1], ylim=[-1, +1], zlim=[1, 3])
plt.show()
