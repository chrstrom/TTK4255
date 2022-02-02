#!/usr/bin/python3

from fnmatch import translate
from turtle import xcor
import numpy as np
import matplotlib.pyplot as plt
from common import *

helicopter_image = plt.imread("../data/quanser.jpg")

K = np.loadtxt("../data/heli_K.txt")
T_from_platform_to_camera = np.loadtxt("../data/platform_to_camera.txt")

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

axis_scale = 0.05
psi = np.deg2rad(11.6)
theta = np.deg2rad(28.9)
phi = np.deg2rad(0)

# 4.2
X_p = np.c_[x0, x1, x2, x3].reshape(4, 4)
X_c = T_from_platform_to_camera @ X_p
u, v = project(K, X_c)

T_from_base_to_platform = T_from_platform_to_camera @ translate_xyz(x=d/2, y=d/2) @ rotate_z(psi) # 4.3
T_from_hinge_to_base = T_from_base_to_platform @ translate_xyz(z=0.325) @ rotate_y(theta) # 4.4
T_from_arm_to_hinge = T_from_hinge_to_base @ translate_xyz(z=-0.05) # 4.5
T_from_rotors_to_arm = T_from_arm_to_hinge @ translate_xyz(x=0.65, z=-0.03) @ rotate_z(phi) # 4.6

marker_points = np.loadtxt("../data/heli_K.txt")

plt.figure()
plt.imshow(helicopter_image)
draw_frame(K, T_from_platform_to_camera, scale=axis_scale)
draw_frame(K, T_from_base_to_platform, scale=axis_scale)
draw_frame(K, T_from_hinge_to_base, scale=axis_scale)
draw_frame(K, T_from_arm_to_hinge, scale=axis_scale)
draw_frame(K, T_from_rotors_to_arm, scale=axis_scale)
plt.scatter(u, v, c="yellow", marker="x", s=100)

#plt.xlim([200, 500])
#plt.ylim([600, 400])
plt.show()
