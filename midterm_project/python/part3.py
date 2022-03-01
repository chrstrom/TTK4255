import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import least_squares

from common import translate, rotate_x, rotate_y, rotate_z, project
import part1b


def T_hat_model_A(kinematic_parameters, rpy):
    T_base_platform = translate(
        kinematic_parameters[0] / 2, kinematic_parameters[0] / 2, 0.0
    ) @ rotate_z(rpy[0])
    T_hinge_base = translate(0.0, 0.0, kinematic_parameters[1]) @ rotate_y(rpy[1])
    T_arm_hinge = translate(0.0, 0.0, -kinematic_parameters[2])
    T_rotors_arm = translate(
        kinematic_parameters[3], 0.0, -kinematic_parameters[4]
    ) @ rotate_x(rpy[2])

    T_base_camera = T_platform_camera @ T_base_platform
    T_hinge_camera = T_base_camera @ T_hinge_base
    T_arm_camera = T_hinge_camera @ T_arm_hinge
    T_rotors_camera = T_arm_camera @ T_rotors_arm

    return T_rotors_camera, T_arm_camera


def u_hat(kinematic_parameters, state):
    marker_points = kinematic_parameters[-3 * M :].reshape((3, M))
    marker_points = np.vstack((marker_points, np.ones((1, M))))

    T_rotors_camera, T_arm_camera = T_hat_model_A(kinematic_parameters, state)

    markers_rotor = T_rotors_camera @ marker_points[:, 3:]
    markers_arm = T_arm_camera @ marker_points[:, :3]

    X = np.hstack([markers_arm, markers_rotor])
    u = project(K, X)
    return u


def residuals(p):
    kinematic_parameters = p[:KP]
    state_parameters = p[KP:]

    r = np.zeros(2 * M * N)
    for i in range(N):
        ui = detections[i, 1::3]
        vi = detections[i, 2::3]
        weights = detections[i, ::3]
        u = np.vstack((ui, vi))

        state = state_parameters[3 * i : 3 * (i + 1)]

        r[2 * M * i : 2 * M * (i + 1)] = (
            weights * (u_hat(kinematic_parameters, state) - u)
        ).flatten()

    return r


if __name__ == "__main__":

    detections = np.loadtxt("../data/detections.txt")
    heli_points = np.loadtxt("../data/heli_points.txt").T
    K = np.loadtxt("../data/K.txt")
    T_platform_camera = np.loadtxt("../data/platform_to_camera.txt")

    initial_lengths = np.array([0.1145, 0.325, 0.050, 0.65, 0.030])
    initial_markers = heli_points[:3, :].flatten()
    initial_kinematic_parameters = np.hstack((initial_lengths, initial_markers))

    initial_state_parameters, _ = part1b.detected_trajectory()
    initial_state_parameters = initial_state_parameters.flatten()
    initial_parameters = np.hstack(
        [initial_kinematic_parameters, initial_state_parameters]
    )

    N = detections.shape[0]
    KP = initial_kinematic_parameters.shape[0]
    M = 7

    A1 = np.ones([2 * M * N, KP])
    B = np.ones([2 * M, 3])
    A2 = B.copy()
    for i in range(N - 1):
        A2 = block_diag(A2, B)
    JS = np.block([A1, A2])

    optimized_parameters = least_squares(
        residuals, initial_parameters, xtol=1e-4, jac_sparsity=JS
    ).x

    image_number = 125
    do_plot = True
    if do_plot:
        kinematic_parameters = optimized_parameters[:KP]
        rpy = optimized_parameters[KP + image_number * 3 : KP + (image_number + 1) * 3]

        plt.imshow(plt.imread("../data/img_sequence/video%04d.jpg" % image_number))
        plt.scatter(
            *u_hat(kinematic_parameters, rpy),
            linewidths=1,
            color="yellow",
            s=10,
            label="LM batch opt. predicted markers"
        )
        plt.legend()
        plt.show()
