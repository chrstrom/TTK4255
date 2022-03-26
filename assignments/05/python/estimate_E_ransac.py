import numpy as np
from estimate_E import *
from epipolar_distance import *
from F_from_E import *


def estimate_E_ransac(xy1, xy2, uv1, uv2, K, distance_threshold, num_trials):
    m = 8
    max_inlier_count = 0
    E_best = -1
    inlier_set_best = 0

    for i in range(num_trials):
        print(f"Iteration {i + 1} of {num_trials}")
        sample = np.random.choice(xy1.shape[1], size=m, replace=False)
        E = estimate_E(xy1[:, sample], xy2[:, sample])
        residuals = epipolar_distance(F_from_E(K, E), uv1, uv2)
        inlier_set = np.abs(residuals) < distance_threshold
        inlier_count = inlier_set.sum()

        if inlier_count > max_inlier_count:
            print(f"new best inlier count of {inlier_count} found at iteration {i}")
            E_best = E
            inlier_set_best = inlier_set
            max_inlier_count = inlier_count

    return E_best, inlier_set_best
