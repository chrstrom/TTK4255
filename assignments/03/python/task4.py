#!/usr/bin/python3

from fnmatch import translate
from turtle import xcor
import numpy as np
import matplotlib.pyplot as plt
import common as cm

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
axis_scale = 0.05
psi = np.deg2rad(11.6)
theta = np.deg2rad(28.9)
phi = np.deg2rad(0)

# 4.2
x0 = np.array([0, 0, 0, 1])
x1 = np.array([d, 0, 0, 1])
x2 = np.array([d, d, 0, 1])
x3 = np.array([0, d, 0, 1])
X_p = np.c_[x0, x1, x2, x3].reshape(4, 4)
X_c = T_from_platform_to_camera @ X_p
u, v = cm.project(K, X_c)

# 4.3
T_from_base_to_camera = (
    T_from_platform_to_camera @ cm.translate_xyz(x=d / 2, y=d / 2) @ cm.rotate_z(psi)
)

# 4.4
T_from_hinge_to_camera = (
    T_from_base_to_camera @ cm.translate_xyz(z=0.325) @ cm.rotate_y(theta)
)

# 4.5
T_from_arm_to_camera = T_from_hinge_to_camera @ cm.translate_xyz(z=-0.05)

# 4.6
T_from_rotors_to_camera = (
    T_from_arm_to_camera @ cm.translate_xyz(x=0.65, z=-0.03) @ cm.rotate_z(phi)
)

# 4.7
marker_points = np.loadtxt("../data/heli_points.txt")
markers_in_arm_frame = marker_points[:3, :]
markers_in_rotor_frame = marker_points[3:, :]

u_arm, v_arm = cm.project(K, T_from_arm_to_camera @ markers_in_arm_frame.T)
u_rotor, v_rotor = cm.project(K, T_from_rotors_to_camera @ markers_in_rotor_frame.T)

plt.figure()
plt.imshow(helicopter_image)
cm.draw_frame(K, T_from_platform_to_camera, scale=axis_scale)
cm.draw_frame(K, T_from_base_to_camera, scale=axis_scale)
cm.draw_frame(K, T_from_hinge_to_camera, scale=axis_scale)
cm.draw_frame(K, T_from_arm_to_camera, scale=axis_scale)
cm.draw_frame(K, T_from_rotors_to_camera, scale=axis_scale)
plt.scatter(u, v, c="yellow", marker="x", s=100)
plt.scatter(u_arm, v_arm, c="yellow", marker=".", s=100)
plt.scatter(u_rotor, v_rotor, c="yellow", marker=".", s=100)
# plt.xlim([200, 500])
# plt.ylim([600, 400])
plt.show()
