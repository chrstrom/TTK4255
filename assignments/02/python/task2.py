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
N_rho = 600
N_theta = 600

###########################################
#
# Task 2.1: Determine appropriate ranges
#
###########################################
Ny, Nx, _ = I_rgb.shape
print(f"image shape: y:{Ny} x:{Nx}")

rho_max = np.sqrt(Ny ** 2 + Nx ** 2)
rho_min = 0
theta_min = -np.pi / 2
theta_max = 3 * np.pi / 2

###########################################
#
# Task 2.2: Compute the accumulator array
#
###########################################
# Zero-initialize an array to hold our votes
H = np.zeros((N_rho, N_theta))

row_from_rho = lambda rho: int(
    np.floor(N_rho * (rho - rho_min) / ((rho_max - rho_min)))
)
col_from_theta = lambda theta: int(
    np.floor(N_theta * (theta - theta_min) / ((theta_max - theta_min)))
)

rho = x * np.cos(theta) + y * np.sin(theta)
for angle, rho in zip(theta, rho):
    row = row_from_rho(rho)
    col = col_from_theta(angle)
    H[row, col] += 1

###########################################
#
# Task 2.3: Extract local maxima
#
###########################################
H_max_x, H_max_y = extract_local_maxima(H, line_threshold)

rho_from_row = lambda row: float((row * (rho_max - rho_min) / N_rho))
theta_from_col = lambda col: float(col * (theta_max - theta_min) / N_theta + theta_min)

# 2) Convert (row, column) back to (rho, theta)r

maxima_theta = []
maxima_rho = []
for x, y in zip(H_max_x, H_max_y):
    maxima_theta.append(theta_from_col(y))
    maxima_rho.append(rho_from_row(x))
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
