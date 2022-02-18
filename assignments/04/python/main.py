#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
from common import *

K = np.loadtxt("../data/K.txt")
detections = np.loadtxt("../data/detections.txt")
XY = np.loadtxt("../data/XY.txt").T
n_total = XY.shape[1]  # Total number of markers (= 24)

fig = plt.figure(figsize=plt.figaspect(0.35))

# for image_number in range(23): # Use this to run on all images
for image_number in [4]:  # Use this to run on a single image
    #  n : Number of successfully detected markers (<= n_total)
    # uv : Pixel coordinates of successfully detected markers
    valid = detections[image_number, 0::3] == True
    uv = np.vstack((detections[image_number, 1::3], detections[image_number, 2::3]))
    uv = uv[:, valid]
    n = uv.shape[1]
    
    # Tip: Helper arrays with 0 and/or 1 appended can be useful if
    # you want to replace for-loops with array/matrix operations.
    # uv1 = np.vstack((uv, np.ones(n)))
    # XY1 = np.vstack((XY, np.ones(n_total)))
    # XY01 = np.vstack((XY, np.zeros(n_total), np.ones(n_total)))
    uv1 = np.vstack((uv, np.ones(n)))
    XY1 = np.vstack((XY, np.ones(n_total)))

    xi_tilde = np.linalg.inv(K) @ uv1
    xi = xi_tilde / xi_tilde[2,:]
    
    H = estimate_H(xi.T, XY1[:, valid].T)
    
    x_tilde_predicted = H @ XY1
    u_tilde_predicted = K @ x_tilde_predicted
    
    u_predicted = u_tilde_predicted / u_tilde_predicted[2,:]
    u_predicted = u_predicted[:2, :]

    error = np.linalg.norm(uv - u_predicted, axis=0)

    print(f"For image {image_number}:")
    print(f"e_min: {np.min(error)}")
    print(f"e_max: {np.max(error)}")
    print(f"e_avg: {np.mean(error)}")
    print("")

    T1, T2 = decompose_H(H) # TASK: Implement this function

    T = T1  # TASK: Choose solution (try both T1 and T2 for Task 3.1, but choose automatically for Task 3.2)

    # The figure should be saved in the data directory as out0000.png, etc.
    # NB! generate_figure expects the predicted pixel coordinates as 'uv_from_H'.
    plt.clf()
    generate_figure(fig, image_number, K, T, uv, u_predicted, XY)
    plt.savefig("../data/out%04d.png" % image_number)
