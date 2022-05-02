#!/usr/bin/env python3

import sympy as sp

from numba import jit
import cv2 as cv
import numpy as np
from match_features import FeatureMatcher
from matlab_inspired_interface import match_features
from hw5 import common
import scipy.optimize as opt

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

def get_jacobian_lambda(K):

    proj_mat = np.hstack((np.eye(3), np.zeros((3,1))))

    # Angles
    phi, theta, psi = sp.symbols("phi, theta, psi")

    # Translations
    tx, ty, tz = sp.symbols("tx, ty, tz")
    X, Y, Z = sp.symbols("X, Y, Z")

    XYZ1 = np.array([X, Y, Z, 1])
    state = [phi, theta, psi, tx, ty, tz]

    T = T_SE3(state)
    uvw = K @ proj_mat @ T @ XYZ1
    uv1 = uvw[:3] / uvw[2]
    u = uv1[0]
    v = uv1[1]

    xy = sp.Matrix([u, v])
    p = sp.Matrix([phi, theta, psi, tx, ty, tz])

    J = xy.jacobian(p)

    return sp.lambdify([phi, theta, psi, tx, ty, tz, X, Y, Z], J, 'numpy')


def solve_NLS_pose(X, K, uv, x0):
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
        self.query = cv.imread(query, cv.IMREAD_GRAYSCALE)
        self.K = np.loadtxt("../data/undistorted/K.txt")

        options = [30000, 4, 0.001, 5, 1.5, 0.9, False]
        self.feature_matcher = FeatureMatcher(options)


    def get_jacobian(self, p, object_points):
        r = object_points.shape[1]
        Jac_lambda = get_jacobian_lambda(self.K)
        J = np.empty((2*r, 6))

        phi, theta, psi, tx, ty, tz = p

        for i in range(r):
            X_point, Y_point, Z_point = object_points[:,i][:3]
            jac_args = [phi, theta, psi, tx, ty, tz, X_point, Y_point, Z_point]
            J[i*2:2*(i+1), :] = Jac_lambda(*jac_args)

        return J


    def run(self, object_points, image_points, K=None):
        time_start = datetime.now()
        if K is None:
            K = self.K


        dist = np.zeros((1, 5))
    
        success, rvecs, tvecs, inliers = cv.solvePnPRansac(object_points, image_points, K, dist)

        x0 = np.zeros((6,))
        x0[3:] = np.reshape(tvecs, (3,))

        p, _ = solve_NLS_pose(object_points, K, image_points, x0)
        #T_hat = common.TSE3(p)

        print(f"Time taken: {datetime.now() - time_start}\n")
        # T_hat[:3, :3] = T_hat[:3, :3].T
        # T_hat[:3, 3] = -
        # T_hat[:3, 3]

        #J = self.get_jacobian(p, object_points)

        #np.savetxt("../localization/Tmq.txt", T_hat)
        return p #, J


    def run_monte_carlo(self, num_trials, sigmas):
        mu_nf = self.K[0, 0]
        mu_ncx = self.K[0, 2]
        mu_ncy = self.K[1, 2]

        sigma_nf = sigmas[0]
        sigma_ncx = sigmas[1]
        sigma_ncy = sigmas[2]

        ps = np.zeros((6, num_trials))

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
        
        for i in range(num_trials):
            print(f"Trial #{i}:")
            nf = np.random.normal(mu_nf, sigma_nf)
            ncx = np.random.normal(mu_ncx, sigma_ncx)
            ncy = np.random.normal(mu_ncy, sigma_ncy)

            K = self.K + np.array([nf, 0, ncx, 0, nf, ncy, 0, 0, 0]).reshape((3, 3))
            ps[:, i] = self.run(object_points, image_points, K)

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