import numpy as np
import matplotlib.pyplot as plt
from common import *
from scipy.optimize import least_squares

# Tip: The solution from HW4 is inside common.py

K = np.loadtxt('../data/K.txt')
u = np.loadtxt('../data/platform_corners_image_minus_one.txt') #each column is (ui, vi).T 
X = np.loadtxt('../data/platform_corners_metric_minus_one.txt') # each column is (X Y 0 1).T
I = plt.imread('../data/img_sequence/video0000.jpg') # Only used for plotting

# Task 2.1 (from HW4)
uv1 = np.vstack((u, np.ones(u.shape[1])))
xy_tilde = np.linalg.inv(K) @ uv1
xy = xy_tilde / xy_tilde[2,:]
xy = xy_tilde[:2, :]
XY = X[:2, :]

H = estimate_H(xy, XY)

# (a)
# XY1 = np.vstack([X[:2, :], X[3, :]])
# T_hat = H
# u_hat = project(K, H@XY1)

# (b)
T1, T2 = decompose_H(H)
translation_z = T1[2, 3]
T_linear = T1 if translation_z < 0 else T2
R0 = np.eye(4)
R0[:3, :3] = T_linear[:3, :3]
# u_hat = project(K, T_hat@X)
# errors = np.linalg.norm(u - u_hat, axis=0)



# Task 2.3
# Parametrized as p0, p1, p2, t0 t1 t2
h = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
def T_from_h(h):
    p = h[:3]
    t = h[3:]
    R = rotate_x(p[0])@rotate_y(p[1])@rotate_z(p[2])@R0

    T = np.eye(4)
    T[:3, :3] = R[:3, :3]
    T[:3, 3] = t
    return T

def resfun(h):
    T = T_from_h(h)
    u_hat = project(K, T@X)
    N = u.shape[0]
    M = u.shape[1]
    r = np.zeros(N*M)
    for i in range(M):
        r[i] = u_hat[0, i] - u[0, i]
        r[i+M] = u_hat[1, i] - u[1, i]

    return r


# Tip: Use the previous image's parameter estimate as initialization
h = least_squares(resfun, x0=h, method='lm').x

T_hat = T_from_h(h)
u_hat = project(K, T_hat@X)
errors = np.linalg.norm(u - u_hat, axis=0)

# Print the reprojection errors requested in Task 2.1 and 2.2.
print('Reprojection error: ')
print('all:', ' '.join(['%.03f' % e for e in errors]))
print('mean: %.03f px' % np.mean(errors))
print('median: %.03f px' % np.median(errors))

plt.imshow(I)
plt.scatter(u[0,:], u[1,:], marker='o', facecolors='white', edgecolors='black', label='Detected')
plt.scatter(u_hat[0,:], u_hat[1,:], marker='.', color='red', label='Predicted')
plt.legend()

# Tip: Draw lines connecting the points for easier understanding
plt.plot(u_hat[0,:], u_hat[1,:], linestyle='--', color='white')

# Tip: To draw a transformation's axes (only requested in Task 2.3)
draw_frame(K, T_hat, scale=0.05, labels=True)

# Tip: To zoom in on the platform:
plt.xlim([200, 500])
plt.ylim([600, 350])

# Tip: To see the entire image:
# plt.xlim([0, I.shape[1]])
# plt.ylim([I.shape[0], 0])

# Tip: To save the figure:
#plt.savefig('out_part2.3_two.png')
print(f"T: {T_hat}")

plt.show()
