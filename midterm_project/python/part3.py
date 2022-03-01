import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import time 

from common import translate, rotate_x, rotate_y, rotate_z, project
import part1b
from plot_all import plot_all

# TODO: plot_all

class KinematicModel:
    def __init__(self):
        self.T_platform_camera = np.loadtxt("../data/platform_to_camera.txt")
        self.detections = np.loadtxt("../data/detections.txt")
        self.K = np.loadtxt("../data/K.txt")
        self.N = self.detections.shape[0]
        self.M = 7
        self.initial_markers = 0.1 * np.ones(21)  # actual values in heli_points[:3, :]

    def jacobian_sparsity(self, KP):
        A1 = np.ones([2 * self.M * self.N, KP])
        A2 = np.kron(np.eye(self.N), np.ones([2 * self.M, 3]))
        return np.block([A1, A2])


class KinematicModelA(KinematicModel):
    def __init__(self):
        super().__init__()
        initial_lengths = 0.1 * np.ones(5)
        self.initial_parameters = np.hstack((initial_lengths, self.initial_markers))
        self.KP = self.initial_parameters.shape[0]
        self.JS = self.jacobian_sparsity(self.KP)

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


class KinematicModelB(KinematicModel):
    def __init__(self):
        super().__init__()
        initial_kinematics = 0.1 * np.ones(18)
        self.initial_parameters = np.hstack((initial_kinematics, self.initial_markers))
        self.KP = self.initial_parameters.shape[0]
        self.JS = self.jacobian_sparsity(self.KP)

    def T_hat(self, kinematic_parameters, rpy):
        """
        2 = arm
        3 = rotor
        kinetic_parameters = [ax12, ay12, az12, lx12, ly12, lz12, markers...]
        """

        ax = kinematic_parameters[:3]
        ay = kinematic_parameters[3:6]
        az = kinematic_parameters[6:9]
        lx = kinematic_parameters[9:12]
        ly = kinematic_parameters[12:15]
        lz = kinematic_parameters[15:18]
        T_1_platform = (
            rotate_x(ax[0])
            @ rotate_y(ay[0])
            @ rotate_z(az[0])
            @ translate(lx[0], ly[0], lz[0])
            @ rotate_z(rpy[0])
        )
        T_1_2 = (
            rotate_x(ax[1])
            @ rotate_y(ay[1])
            @ rotate_z(az[1])
            @ translate(lx[1], ly[1], lz[1])
            @ rotate_y(rpy[1])
        )
        T_2_3 = (
            rotate_x(ax[2])
            @ rotate_y(ay[2])
            @ rotate_z(az[2])
            @ translate(lx[2], ly[2], lz[2])
            @ rotate_x(rpy[2])
        )

        T_1_camera = self.T_platform_camera @ T_1_platform
        T_2_camera = T_1_camera @ T_1_2
        T_3_camera = T_2_camera @ T_2_3

        return T_3_camera, T_2_camera


class KinematicModelC(KinematicModel):
    def __init__(self):
        super().__init__()
        initial_kinematics = 0.1 * np.ones(12)
        self.initial_parameters = np.hstack((initial_kinematics, self.initial_markers))
        self.KP = self.initial_parameters.shape[0]
        self.JS = self.jacobian_sparsity(self.KP)

    def T_hat(self, kinematic_parameters, rpy):
        """
        2 = arm
        3 = rotor
        kinetic_parameters = [ax12, ay12, az12, lx12, ly12, lz12, markers...]
        """

        ax = kinematic_parameters[:2]
        ay = kinematic_parameters[2:4]
        az = kinematic_parameters[4:6]
        lx = kinematic_parameters[6:8]
        ly = kinematic_parameters[8:10]
        lz = kinematic_parameters[10:12]
        T_1_platform = (
            rotate_x(ax[0])
            @ rotate_y(ay[0])
            @ translate(lx[0], ly[0], 0.0)
            @ rotate_z(rpy[0])
        )
        T_1_2 = (
            rotate_x(ax[1])
            @ rotate_z(az[0])
            @ translate(lx[1], 0.0, lz[0])
            @ rotate_y(rpy[1])
        )
        T_2_3 = (
            rotate_y(ay[1])
            @ rotate_z(az[1])
            @ translate(0.0, ly[1], lz[1])
            @ rotate_x(rpy[2])
        )

        T_1_camera = self.T_platform_camera @ T_1_platform
        T_2_camera = T_1_camera @ T_1_2
        T_3_camera = T_2_camera @ T_2_3

        return T_3_camera, T_2_camera


class BatchOptimizer:
    def __init__(self, model, tol=1e-6, verbosity=0):
        self.model = model
        self.xtol = tol
        self.verbosity = verbosity

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
            xtol=self.xtol,
            jac_sparsity=self.model.JS,
            verbose=self.verbosity,
        )

        return optimized_parameters.x


if __name__ == "__main__":

    kinematic_model = KinematicModelA()

    tol = 1e-8
    optimizer = BatchOptimizer(kinematic_model, tol=tol, verbosity=2)

    time_start = time.time()
    optimized_parameters = optimizer.optimize()
    time_end = time.time()

    print("Time taken to run batch optimization with tolerance {:f}: {:.4f} seconds.".format(tol, time_end - time_start))
    do_plot = True
    if do_plot:
        all_r = []
        all_p = []
        KP = kinematic_model.KP
        for image_number in range(kinematic_model.N):
            rpy = optimized_parameters[
                KP + 3 * image_number : KP + 3 * (image_number + 1)
            ]
            all_p.append(rpy)
        all_r = optimizer.residuals(optimized_parameters).reshape((351, 14))
        all_p = np.array(all_p)

        all_detections = np.loadtxt("../data/detections.txt")
        plot_all(all_p, all_r, all_detections, subtract_initial_offset=True)
        # plt.savefig('out_part3.png')
        plt.show()

    do_anim = False
    if do_anim:
        plt.ion()
        fig, ax = plt.subplots()
        plt.draw()
        KP = kinematic_model.KP
        for image_number in range(kinematic_model.N):
            kinematic_parameters = optimized_parameters[:KP]
            rpy = optimized_parameters[
                KP + 3 * image_number : KP + 3 * (image_number + 1)
            ]

            plt.imshow(plt.imread("../data/img_sequence/video%04d.jpg" % image_number))
            ax.scatter(
                *optimizer.u_hat(kinematic_parameters, rpy),
                linewidths=1,
                color="yellow",
                s=10,
            )

            plt.pause(0.05)
            ax.clear()
