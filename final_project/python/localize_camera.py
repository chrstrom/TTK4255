#!/usr/bin/env python3

from operator import index
import cv2 as cv
import numpy as np
from match_features import FeatureMatcher
from matlab_inspired_interface import match_features

class LocalizeCamera():
    """
    Goal: localize camera in arbitrary image
    Use show_localization_results.py to test
    """

    def __init__(self, query, model):
        
        self.X = model[0:3, :].T
        self.model_desc = np.load("../localization/desc.npy")
        self.query = cv.imread(query, cv.IMREAD_GRAYSCALE)
        self.K = np.loadtxt("../data/calibration/K.txt")

        options = [30000, 4, 0.001, 5, 1.5, 0.9, True]
        self.feature_matcher = FeatureMatcher(options)

    def run(self):

        keypoints, desc = self.feature_matcher.get_keypoints(self.query)
        index_pairs, match_metric = match_features(desc, self.model_desc, 0.9, True)

        N = min(keypoints.shape[0], self.X.shape[0], index_pairs.shape[0])

        kp_query = np.zeros((N, 2))
        kp_model = np.zeros((N, 3))

        for i in range(N):
            kp_query[i, :] = keypoints[index_pairs[i, 0]]
            kp_model[i, :] = self.X[index_pairs[i, 1]]


        _, R_vec, t, inliers = cv.solvePnPRansac(kp_model, kp_query, self.K, np.zeros((1, 5)))

        R, _ = cv.Rodrigues(R_vec)
        t = t[:, 0]

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, -1] = t
        
        np.savetxt("../localization/Tmq.txt", T.T)


if __name__ == "__main__":


    model = np.load("../localization/X.npy")

    query = "../data/undistorted/IMG_8210.jpg"

    localize = LocalizeCamera(query, model)
    localize.run()