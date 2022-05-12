#!/usr/bin/env python3

import sympy as sp

from numba import jit
import cv2 as cv
import numpy as np
from match_features import FeatureMatcher
from matlab_inspired_interface import match_features
from hw5 import common
import scipy.optimize as opt
import matplotlib.pyplot as plt

from datetime import datetime



def T_SE3 (params):
    phi, theta, psi = params[0], params[1], params[2]
    translation = np.array(params[3:])

    Rx = np.array([
        [1,     0,          0  ],
        [0, sp.cos(phi), -sp.sin(phi)],
        [0, sp.sin(phi), sp.cos(phi)]
    ])
    Ry = np.array([
        [sp.cos(theta),  0, sp.sin(theta)],
        [0,           1,     0     ],
        [-sp.sin(theta), 0, sp.cos(theta)]
    ])
    Rz = np.array([
        [sp.cos(psi), -sp.sin(psi),  0],
        [sp.sin(psi), sp.cos(psi),   0],
        [0,         0,         1]
    ])
    
    R = Rx @ Ry @ Rz

    t = np.reshape(translation, (3, 1))

    T = np.block([
        [       R,          t],
        [np.zeros((1,3)),   1]
    ])

    return T


def nls_pose_estimator(X, K, uv, x0):
    X1 = np.c_[X, np.ones((X.shape[0], 1))]
    uv1 = np.c_[uv, np.ones((uv.shape[0], 1))]

    @jit(nopython=True)
    def model(XYZ1, K, T_SE3):
        """
        Takes in world points, camera matrix and an SE(3) matrix and projects points onto image plane
        """
        proj_mat = np.hstack((np.eye(3), np.zeros((3,1))))
        xyw = proj_mat @ T_SE3 @ XYZ1.T
        xy1 = xyw[:3, :] / xyw[2,:]
        
        uv1 = K @ xy1
        return uv1

    def resfun(theta):
        TSE3 = common.TSE3(theta)
        uv2_hat = model(X1, K, TSE3)
        diffs = uv2_hat - uv1.T 

        residuals = np.hstack((diffs[0,:], diffs[1, :]))
        return residuals
    
    result = opt.least_squares(resfun, x0, diff_step=1e-8)
    p = result.x
    jac = result.jac

    return p, jac

class LocalizeCamera():
    """
    Goal: localize camera in arbitrary image
    Use show_localization_results.py to test
    """

    def __init__(self, query, model):
        
        self.object_points = model[0].T[:, :3]

        self.model_desc = model[1]
        self.query = plt.imread(query)
        self.K = np.loadtxt("../data/undistorted/K.txt")

        options = [30000, 4, 0.001, 5, 1.5, 0.9, False]
        self.feature_matcher = FeatureMatcher(options)

    def get_matches(self):
        print("  Detecting keypoints...")
        keypoints, desc = self.feature_matcher.get_keypoints(self.query)
        print("    Done!")

        print("  Matching features...")
        index_pairs, match_metric = match_features(
            desc, self.model_desc, 0.9, True
        )

        print(f"    Done! Found {index_pairs.shape[0]} matching features")

        image_points = keypoints[index_pairs[:, 0]]
        object_points = self.object_points[index_pairs[:, 1]]

        return image_points, object_points

    def get_colors(self):
        uv1_in = np.loadtxt("../localization/uv1_in_ivan.out")
        colors = self.query[uv1_in[1,:].astype(np.int32), uv1_in[0,:].astype(np.int32), :]

        return colors


    def run(self, K=None):
        time_start = datetime.now()
        if K is None:
            K = self.K

        image_points, object_points = self.get_matches()

        success, rvecs, tvecs, inliers = cv.solvePnPRansac(object_points, image_points, K, np.zeros((1, 5)))

        x0 = np.hstack((rvecs.reshape((3,)), tvecs.reshape((3,))))
        p, J = nls_pose_estimator(object_points, K, image_points, x0)
        T_hat = common.TSE3(p)

        print(f"Time taken: {datetime.now() - time_start}\n")

        return p, J, T_hat


    def run_monte_carlo(self, num_trials, sigmas):
        mu_nf = self.K[0, 0]
        mu_ncx = self.K[0, 2]
        mu_ncy = self.K[1, 2]

        sigma_nf = sigmas[0]
        sigma_ncx = sigmas[1]
        sigma_ncy = sigmas[2]

        ps = np.zeros((6, num_trials))

        image_points, object_points = self.get_matches()
        
        for i in range(num_trials):
            print(f"Trial #{i}:")
            nf = np.random.normal(mu_nf, sigma_nf)
            ncx = np.random.normal(mu_ncx, sigma_ncx)
            ncy = np.random.normal(mu_ncy, sigma_ncy)

            K = self.K + np.array([nf, 0, ncx, 0, nf, ncy, 0, 0, 0]).reshape((3, 3))
            ps[:, i], _, _ = self.run(object_points, image_points, K)

        # Calculate sample var (row-wise)
        # Get means for each row
        means = [row.mean() for row in ps]

        # Calculate squared errors
        squared_errors = [(row-mean)**2 for row, mean in zip(ps, means)]

        # Calculate "mean for each row of squared errors" (aka the variance)
        variances = [row.mean() for row in squared_errors]

        return variances



if __name__ == "__main__":
    pass

    # model = np.load("../localization/X.npy")

    # query = "../data/undistorted/IMG_8227.jpg"

    # localize = LocalizeCamera(query, model)
    # localize.run()