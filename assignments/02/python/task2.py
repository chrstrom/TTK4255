#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from common import *

# This bit of code is from HW1.
edge_threshold = 0.015
blur_sigma = 1
filename = "../data/grid.jpg"
I_rgb = plt.imread(filename)
I_rgb = im2double(
    I_rgb
)  # Ensures that the image is in floating-point with pixel values in [0,1].
I_gray = rgb_to_gray(I_rgb)
Ix, Iy, Im = derivative_of_gaussian(I_gray, sigma=blur_sigma)  # See HW1 Task 3.6
x, y, theta = extract_edges(Ix, Iy, Im, edge_threshold)

# You can adjust these for better results
line_threshold = 0.175
N_rho = 400
N_theta = 400

###########################################
#
# Task 2.1: Determine appropriate ranges
#
###########################################
Ny, Nx, _ = I_rgb.shape
print(f"image shape: y:{Ny} x:{Nx}")

rho_max = np.sqrt(Ny ** 2 + Nx ** 2)
rho_min = -rho_max
theta_min = np.pi
theta_max = -theta_min

###########################################
#
# Task 2.2: Compute the accumulator array
#
###########################################
# Zero-initialize an array to hold our votes
H = np.zeros((N_rho, N_theta))

rho = x * np.cos(theta) + y * np.sin(theta)
rows_from_rho = np.floor(N_rho * (rho - rho_min) / ((rho_max - rho_min))).astype(int)
cols_from_theta = np.floor(
    N_theta * (theta - theta_min) / ((theta_max - theta_min))
).astype(int)

for row, col in zip(rows_from_rho, cols_from_theta):
    H[row, col] += 1

###########################################
#
# Task 2.3: Extract local maxima
#
###########################################
H_max_x, H_max_y = extract_local_maxima(H, line_threshold)

# 2) Convert (row, column) back to (rho, theta)r
maxima_rho = (H_max_x * (rho_max - rho_min) / N_rho + rho_min).astype(float)
maxima_theta = (H_max_y * (theta_max - theta_min) / N_theta + theta_min).astype(float)

###########################################
#
# Figure 2.2: Display the accumulator array and local maxima
#
###########################################
plt.figure()
plt.imshow(H, extent=[theta_min, theta_max, rho_max, rho_min], aspect="auto")
plt.colorbar(label="Votes")
plt.scatter(maxima_theta, maxima_rho, marker=".", color="red")
plt.title("Accumulator array")
plt.xlabel("$\\theta$ (radians)")
plt.ylabel("$\\rho$ (pixels)")
# plt.savefig('out_array.png', bbox_inches='tight', pad_inches=0) # Uncomment to save figure

###########################################
#
# Figure 2.3: Draw the lines back onto the input image
#
###########################################
plt.figure()
plt.imshow(I_rgb)
plt.xlim([0, I_rgb.shape[1]])
plt.ylim([I_rgb.shape[0], 0])
for theta, rho in zip(maxima_theta, maxima_rho):
    draw_line(theta, rho, color="yellow")
plt.title("Dominant lines")
# plt.savefig('out_lines.png', bbox_inches='tight', pad_inches=0) # Uncomment to save figure

plt.show()
