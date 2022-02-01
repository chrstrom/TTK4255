#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from common import *

K = np.loadtxt("../data/task2K.txt")
X = np.loadtxt("../data/task2points.txt")

u, v = project(K, X)

width, height = 600, 400

plt.figure(figsize=(4, 3))
plt.scatter(u, v, c="black", marker=".", s=20)

plt.axis("image")
plt.xlim([0, width])
plt.ylim([height, 0])
# plt.savefig('plots/task3-2.png', bbox_inches='tight', pad_inches=0) # Uncomment to save figure
plt.show()
