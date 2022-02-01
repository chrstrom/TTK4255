#!/usr/bin/python3

from turtle import xcor
import numpy as np
import matplotlib.pyplot as plt
from common import *

helicopter_image = plt.imread("../data/quanser.jpg")

K = np.loadtxt("../data/heli_K.txt")
T_pc = np.loadtxt("../data/platform_to_camera.txt")

# Screw layout:
#  x3 ---- x2
#  |        |
#  |        |
#  x0 ---- x1
# Origin is in x0, +x towards x1 and +y towards x3
# All screws are placed with a distance d from the others

d = 0.1145
x0 = np.array([0, 0, 0, 1])
x1 = np.array([d, 0, 0, 1])
x2 = np.array([d, d, 0, 1])
x3 = np.array([0, d, 0, 1])

X_p = np.c_[x0, x1, x2, x3].reshape(4, 4)
X_c = T_pc @ X_p
u, v = project(K, X_c)

plt.figure()
plt.imshow(helicopter_image)
draw_frame(K, T_pc, scale=0.05)
plt.scatter(u, v, c="white", marker="x", s=100)

plt.xlim([200, 500])
plt.ylim([600, 400])
plt.show()
