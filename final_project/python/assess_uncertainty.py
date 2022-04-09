"""
generating a few (e.g. ten) hypothetical outcomes from the
distribution defined by the standard deviations, and undistorting an image with each set of hypothetical
parameters. The resulting images can then be compared, to get a visual impression of the uncertainty.
"""


# Step 1: Select an image to use
# Step 2: get mean and variance of distortion coefficients
# Step 3: Make gaussians from values in 2
# Step 4: N times: sample from the distributions in 3 and cv.undistort image, save image to file.

"""
Distortion coefficients
--------------------------------
k1:     -0.06652 +/- 0.00109
k2:      0.06534 +/- 0.00624
k3:     -0.07555 +/- 0.01126
p1:      0.00065 +/- 0.00011
p2:     -0.00419 +/- 0.00014
"""


import numpy as np
import cv2 as cv

if __name__ == '__main__':
    N = 10

    image = "../data/calibration/checkerboard.png"
    

    camera_matrix = np.array(((2359.40946, 0, 1370.05852), (0, 2359.61091, 1059.63818), (0, 0, 1)))

    for i in range(N):
            
        k1 = np.random.normal(-0.06652, 0.00109)
        k2 = np.random.normal(0.06534, 0.00624)
        k3 = np.random.normal(-0.07555, 0.01126)
        p1 = np.random.normal(0.00065, 0.00011)
        p2 = np.random.normal(-0.00419, 0.00014)

        distortion_coefficients = np.array((k1, k2, k3, p1, p2))

        I = cv.imread(image, cv.IMREAD_GRAYSCALE)

        dst = cv.undistort(I, camera_matrix, distortion_coefficients, camera_matrix)

        # cv.imshow("distorted image with sampled distortion coefficients", dst)
        # cv.waitKey()

        cv.imwrite(f"../data/distortion_sample/sample{i:03d}.png", dst) 



