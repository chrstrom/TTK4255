import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import least_squares

from common import translate, rotate_x, rotate_y, rotate_z, project
import part1b


class KinematicModelA:
    def __init__(self):

        self.T_platform_camera = np.loadtxt("../data/platform_to_camera.txt")
        self.detections = np.loadtxt("../data/detections.txt")
        self.K = np.loadtxt("../data/K.txt")

        self.N = self.detections.shape[0]
        self.M = 7

        initial_lengths = np.array(
            [0.1, 0.1, 0.1, 0.1, 0.1]
        )  # actual values: 0.1145, 0.325, 0.050, 0.65, 0.030
        initial_markers = 0.1 * np.ones(21)  # actual values in heli_points[:3, :]
        self.initial_parameters = np.hstack((initial_lengths, initial_markers))
        self.KP = self.initial_parameters.shape[0]

        A1 = np.ones([2 * self.M * self.N, self.KP])
        B = np.ones([2 * self.M, 3])
        A2 = B.copy()
        for _ in range(self.N - 1):
            A2 = block_diag(A2, B)
        self.JS = np.block([A1, A2])

    def T_hat(self, kinematic_parameters, rpy):
        T_base_platform = translate(
            kinematic_parameters[0] / 2, kinematic_parameters[0] / 2, 0.0
        ) @ rotate_z(rpy[0])
        T_hinge_base = translate(0.0, 0.0, kinematic_parameters[1]) @ rotate_y(rpy[1])
        T_arm_hinge = translate(0.0, 0.0, -kinematic_parameters[2])
        T_rotors_arm = translate(
            kinematic_parameters[3], 0.0, -kinematic_parameters[4]
        ) @ rotate_x(rpy[2])

        T_base_camera = self.T_platform_camera @ T_base_platform
        T_hinge_camera = T_base_camera @ T_hinge_base
        T_arm_camera = T_hinge_camera @ T_arm_hinge
        T_rotors_camera = T_arm_camera @ T_rotors_arm

        return T_rotors_camera, T_arm_camera


class BatchOptimizer:
    def __init__(self, model):
        self.model = model

        initial_state_parameters, _ = part1b.detected_trajectory()
        initial_state_parameters = initial_state_parameters.flatten()
        self.initial_parameters = np.hstack(
            [kinematic_model.initial_parameters, initial_state_parameters]
        )

    def u_hat(self, kinematic_parameters, state):
        marker_points = kinematic_parameters[-3 * self.model.M :].reshape(
            (3, self.model.M)
        )
        marker_points = np.vstack((marker_points, np.ones((1, self.model.M))))

        T_rotors_camera, T_arm_camera = self.model.T_hat(kinematic_parameters, state)

        markers_rotor = T_rotors_camera @ marker_points[:, 3:]
        markers_arm = T_arm_camera @ marker_points[:, :3]

        X = np.hstack([markers_arm, markers_rotor])
        u = project(self.model.K, X)
        return u

    def residuals(self, p):
        kinematic_parameters = p[: self.model.KP]
        state_parameters = p[self.model.KP :]

        r = np.zeros(2 * self.model.M * self.model.N)
        for i in range(self.model.N):
            ui = self.model.detections[i, 1::3]
            vi = self.model.detections[i, 2::3]
            weights = self.model.detections[i, ::3]
            u = np.vstack((ui, vi))

            state = state_parameters[3 * i : 3 * (i + 1)]

            r[2 * self.model.M * i : 2 * self.model.M * (i + 1)] = (
                weights * (self.u_hat(kinematic_parameters, state) - u)
            ).flatten()

        return r

    def optimize(self):

        optimized_parameters = least_squares(
            self.residuals,
            self.initial_parameters,
            xtol=1e-6,
            jac_sparsity=self.model.JS,
        )

        return optimized_parameters.x


if __name__ == "__main__":

    kinematic_model = KinematicModelA()

    optimizer = BatchOptimizer(kinematic_model)
    optimized_parameters = optimizer.optimize()

    image_number = 0
    do_plot = True
    if do_plot:
        KP = kinematic_model.KP
        kinematic_parameters = optimized_parameters[:KP]
        rpy = optimized_parameters[KP + image_number * 3 : KP + (image_number + 1) * 3]

        plt.imshow(plt.imread("../data/img_sequence/video%04d.jpg" % image_number))
        plt.scatter(
            *optimizer.u_hat(kinematic_parameters, rpy),
            linewidths=1,
            color="yellow",
            s=10,
            label="LM batch opt. predicted markers"
        )
        plt.legend()
        plt.show()
