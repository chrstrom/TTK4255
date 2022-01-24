#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
from common import *
from scipy import ndimage

# Note: the sample image is naturally grayscale
I = rgb_to_gray(im2double(plt.imread("../data/calibration.jpg")))

###########################################
#
# Task 3.1: Compute the Harris-Stephens measure
#
###########################################
sigma_D = 1
sigma_I = 3
alpha = 0.06

Ix, Iy, Im = derivative_of_gaussian(I, sigma_D)

Axx = gaussian(Ix**2, sigma_I)
Ayy = gaussian(Iy**2, sigma_I)
Axy = gaussian(Ix*Iy, sigma_I)
Ayx = gaussian(Iy*Ix, sigma_I)
assert np.array_equal(Axy, Ayx)

# A = np.block([[Axx, Axy], [Ayx, Ayy]])
# print(A.shape)
# A is not square, how to calculate determinant???
det = Axx*Ayy - (Axy + Ayx)
trace = Axx + Ayy
harris_measure = det - alpha*trace**2

###########################################
#
# Task 3.4: Extract local maxima
#
###########################################
threshold = 0.001
corners_y, corners_x = extract_local_maxima(harris_measure, threshold)

###########################################
#
# Figure 3.1: Display Harris-Stephens corner strength
#
###########################################
plt.figure(figsize=(13, 5))
plt.imshow(harris_measure)
plt.colorbar(label="Corner strength")
plt.tight_layout()
# plt.savefig('out_corner_strength.png', bbox_inches='tight', pad_inches=0) # Uncomment to save figure in working directory

###########################################
#
# Figure 3.4: Display extracted corners
#
###########################################
plt.figure(figsize=(10, 5))
plt.imshow(I, cmap="gray")
plt.scatter(corners_x, corners_y, linewidths=1, edgecolor="black", color="yellow", s=9)
plt.tight_layout()
# plt.savefig('out_corners.png', bbox_inches='tight', pad_inches=0) # Uncomment to save figure in working directory

plt.show()
