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
    pass
