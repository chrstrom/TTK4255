#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from matlab_inspired_interface import match_features, show_matched_features

from hw5.figures import *
from hw5.estimate_E import *
from hw5.decompose_E import *
from hw5.triangulate_many import *
from hw5.epipolar_distance import *
from hw5.estimate_E_ransac import *


class RANSAC:
    def __init__(self):
        self.distance_threshold = 0.75
        inlier_fraction = 0.5
        success_probability = 0.99
        self.num_trials = int(
            np.ceil(np.log(1 - success_probability) / np.log(1 - inlier_fraction ** 8))
        )

    def run(self, xy1, xy2, uv1, uv2, K):

        inlier_set = estimate_E_ransac(
            xy1, xy2, uv1, uv2, K, self.distance_threshold, self.num_trials
        )

        uv1_in = uv1[:, inlier_set]
        uv2_in = uv2[:, inlier_set]
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

        return X, E, [xy1_in, xy2_in], [uv1_in, uv2_in]


class FeatureMatcher:
    def __init__(self, image_one, image_two, opts):
        self.I1 = cv.imread(image_one, cv.IMREAD_GRAYSCALE)
        self.I2 = cv.imread(image_two, cv.IMREAD_GRAYSCALE)

        self.I1P = plt.imread(image_one) / 255.0
        self.I2P = plt.imread(image_two) / 255.0

        self.nfeatures = opts[0]
        self.nOctaveLayers = opts[1]
        self.contrastThreshold = opts[2]
        self.edgeThreshold = opts[3]
        self.sigma = opts[4]
        self.max_ratio = opts[5]
        self.unique = opts[6]

        self.K = np.loadtxt("/home/strom/TTK4255/final_project/data/calibration/K.txt")
        self.K_inv = np.linalg.inv(self.K)

        self.ransac = RANSAC()

    def __project(self, X):
        K_inv = np.linalg.inv(self.K)
        X_tilde = K_inv @ X
        return X_tilde[:2, :] / X_tilde[2, :]

    def __get_matches(self, kp1, kp2, index_pairs):

        kp1 = kp1[index_pairs[:, 0]]
        kp2 = kp2[index_pairs[:, 1]]

        matches = np.zeros((kp1.shape[0], 4))

        for i in range(min(kp1.shape[0], kp2.shape[0])):
            matches[i, 0] = kp1[i][0]
            matches[i, 1] = kp1[i][1]
            matches[i, 2] = kp2[i][0]
            matches[i, 3] = kp2[i][1]

        return matches

    def assess_n_best_features(self, N, index_pairs, match_metric, kp1, kp2):
        best_index_pairs = index_pairs[np.argsort(match_metric)[:N]]
        best_kp1 = kp1[best_index_pairs[:, 0]]
        best_kp2 = kp2[best_index_pairs[:, 1]]

        plt.figure()
        plt.title("Best 50 matched features between IMG_8221 and IMG_8223")
        show_matched_features(self.I1, self.I2, best_kp1, best_kp2, method="falsecolor")
        plt.figure()
        plt.title("Best 50 matched features between IMG_8221 and IMG_8223")
        show_matched_features(self.I1, self.I2, best_kp1, best_kp2, method="montage")
        plt.show()

    def run(self, do_assessment=False):
        print("Detecting keypoints...")
        sift = cv.SIFT_create(
            self.nfeatures,
            self.nOctaveLayers,
            self.contrastThreshold,
            self.edgeThreshold,
            self.sigma,
        )
        kp1, desc1 = sift.detectAndCompute(self.I1, None)
        kp2, desc2 = sift.detectAndCompute(self.I2, None)
        kp1 = np.array([kp.pt for kp in kp1])
        kp2 = np.array([kp.pt for kp in kp2])
        print("  Done!")

        print("Matching features...")
        index_pairs, match_metric = match_features(
            desc1, desc2, self.max_ratio, self.unique
        )
        matches = self.__get_matches(kp1, kp2, index_pairs)
        print("  Done! Found %d matches" % index_pairs.shape[0])

        print("Robustifying with RANSAC...")
        uv1 = np.vstack([matches[:, :2].T, np.ones(matches.shape[0])])
        uv2 = np.vstack([matches[:, 2:4].T, np.ones(matches.shape[0])])
        xy1 = self.__project(uv1)
        xy2 = self.__project(uv2)
        X, E, _, uv_inliers = self.ransac.run(xy1, xy2, uv1, uv2, self.K)
        F = self.K_inv.T @ E @ self.K_inv
        print("  Done! Drawing point cloud...")

        draw_point_cloud(
            X, self.I1P, uv_inliers[0], xlim=[-4, +4], ylim=[-4, +4], zlim=[2, 8]
        )

        plt.show()
        if do_assessment:
            print("Assess performance...")
            draw_correspondences(
                self.I1P, self.I2P, uv_inliers[0], uv_inliers[1], F, sample_size=8
            )
            plt.show()
            self.assess_n_best_features(50, index_pairs, match_metric, kp1, kp2)


if __name__ == "__main__":

    image_one = "../data/IMG_8221.jpg"
    image_two = "../data/IMG_8223.jpg"

    # You will want to pass other options to SIFT_create. See the documentation:
    # https://docs.opencv.org/4.x/d7/d60/classcv_1_1SIFT.html

    # NB! You will want to experiment with different options for the ratio test and
    # "unique" (cross-check).

    # nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma, max_ratio, unique
    options = [30000, 4, 0.001, 5, 1.5, 0.9, False]

    feature_matcher = FeatureMatcher(image_one, image_two, options)

    feature_matcher.run()
