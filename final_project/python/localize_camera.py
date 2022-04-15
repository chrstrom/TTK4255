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
        
        self.object_points = model.T[:, :3]
        self.model_desc = np.load("../localization/desc.npy")
        self.query = cv.imread(query, cv.IMREAD_GRAYSCALE)
        self.K = np.loadtxt("../data/undistorted/K.txt")

        options = [30000, 4, 0.001, 5, 1.5, 0.9, False]
        self.feature_matcher = FeatureMatcher(options)


    def run(self):

        print("Detecting keypoints...")
        keypoints, desc = self.feature_matcher.get_keypoints(self.query)
        print("  Done!")

        print("Matching features...")
        index_pairs, match_metric = match_features(
            desc, self.model_desc, 0.9, True
        )

        print(f"  Done! Found {index_pairs.shape[0]} matching features")

        image_points = keypoints[index_pairs[:, 0]]
        object_points = self.object_points[index_pairs[:, 1]]

        dist = np.zeros((1, 5))
    
        success, rvecs, tvecs, inliers = cv.solvePnPRansac(object_points, image_points, self.K, dist, reprojectionError=100)

        if not success:
            print("solvePnPRansac did not succeed...") 
            exit()

        print(rvecs.shape)
        print(tvecs.shape)
        print(inliers.shape)   
    
        T_hat = np.eye(4)
        R, _ = cv.Rodrigues(rvecs)

        T_hat[:3, :3] = R.T
        T_hat[:3, 3] = tvecs[:, 0]

        print(T_hat)



        np.savetxt("../localization/Tmq.txt", T_hat.T)


if __name__ == "__main__":


    model = np.load("../localization/X.npy")

    query = "../data/undistorted/IMG_8227.jpg"

    localize = LocalizeCamera(query, model)
    localize.run()