#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from common import *

K = np.loadtxt("../data/task2K.txt")
X_o = np.loadtxt("../data/task3points.txt")

T_co = translate_xyz(z=6) @ rotate_x(np.deg2rad(15)) @ rotate_y(np.deg2rad(45))
X_c = T_co @ X_o

u, v = project(K, X_c)

width, height = 600, 400

plt.figure(figsize=(4, 3))
plt.scatter(u, v, c="black", marker=".", s=20)

draw_frame(K, T_co)

plt.axis("image")
plt.xlim([0, width])
plt.ylim([height, 0])
# plt.savefig('plots/task3-2.png', bbox_inches='tight', pad_inches=0) # Uncomment to save figure
plt.show()
