from operator import index
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from matlab_inspired_interface import match_features, show_matched_features
from datetime import datetime

from hw5.figures import *
from hw5.estimate_E import *
from hw5.decompose_E import *
from hw5.triangulate_many import *
from hw5.epipolar_distance import *
from hw5.estimate_E_ransac import *
from datetime import datetime

I1 = cv.imread("../data/IMG_8221.jpg", cv.IMREAD_GRAYSCALE)
I2 = cv.imread("../data/IMG_8223.jpg", cv.IMREAD_GRAYSCALE)

# You will want to pass other options to SIFT_create. See the documentation:
# https://docs.opencv.org/4.x/d7/d60/classcv_1_1SIFT.html

sift = cv.SIFT_create(nfeatures=3000)
kp1, desc1 = sift.detectAndCompute(I1, None)
kp2, desc2 = sift.detectAndCompute(I2, None)
kp1 = np.array([kp.pt for kp in kp1])
kp2 = np.array([kp.pt for kp in kp2])

# NB! You will want to experiment with different options for the ratio test and
# "unique" (cross-check).
index_pairs, match_metric = match_features(desc1, desc2, max_ratio=0.9, unique=False)
print(index_pairs[:10])
print("Found %d matches" % index_pairs.shape[0])

# Plot the 50 best matches in two ways
best_index_pairs = index_pairs[np.argsort(match_metric)[:50]]
best_kp1 = kp1[best_index_pairs[:, 0]]
best_kp2 = kp2[best_index_pairs[:, 1]]
plt.figure()
plt.title("Best 50 matched features between IMG_8221 and IMG_8223")
show_matched_features(I1, I2, best_kp1, best_kp2, method="falsecolor")
plt.figure()
plt.title("Best 50 matched features between IMG_8221 and IMG_8223")
show_matched_features(I1, I2, best_kp1, best_kp2, method="montage")
plt.show()


def project(K, X):
    X_tilde = K @ X
    return X_tilde[:2, :] / X_tilde[2, :]


K = np.loadtxt("/home/strom/TTK4255/final_project/data/calibration/K.txt")

K_inv = np.linalg.inv(K)

kp1 = kp1[index_pairs]
kp2 = kp2[index_pairs]
matches = []

for i in range(kp1.shape[0]):
    matches.append([kp1[i, 0], kp1[i, 1], kp2[i, 0], kp2[i, 1]])

print(matches.shape)

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
