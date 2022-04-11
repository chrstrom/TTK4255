#!/usr/bin/env python3

import numpy as np
import cv2 as cv
import glob
from os.path import join, basename, realpath, dirname, exists, splitext


class CameraCalibration:
    def __init__(self):

        self.image_path_pattern = "../data/calibration/*.jpg"
        self.output_folder = dirname(self.image_path_pattern)

        self.board_size = (
            7,
            4,
        )  # Number of internal corners of the checkerboard (see tutorial)
        self.square_size = (
            10  # Real world length of the sides of the squares (see HW6 Task 1.5)
        )

        self.calibrate_flags = (
            0  # Use default settings (three radial and two tangential)
        )
        # self.calibrate_flags = cv.CALIB_ZERO_TANGENT_DIST|cv.CALIB_FIX_K3 # Disable tangential distortion and third radial distortion coefficient

        # Flags to findChessboardCorners that improve performance
        self.detect_flags = (
            cv.CALIB_CB_ADAPTIVE_THRESH
            + cv.CALIB_CB_NORMALIZE_IMAGE
            + cv.CALIB_CB_FAST_CHECK
        )

        # Termination criteria for cornerSubPix routine
        self.subpix_criteria = (
            cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )

    def __save_np(self, data, file):
        np.save(join(self.output_folder, file), data)

    def __load_np(self, file):
        return np.load(join(self.output_folder, file))

    def __save_txt(self, data, file):
        np.savetxt(join(self.output_folder, file), data)

    def __load_txt(self, file):
        return np.loadtxt(join(self.output_folder, file)).astype(np.int32)

    def save_calibration_results(self, results, X_all, u_all):
        _, K, dc, rvecs, tvecs, std_int, _, _ = results

        mean_errors = []
        for i in range(len(X_all)):
            u_hat, _ = cv.projectPoints(X_all[i], rvecs[i], tvecs[i], K, dc)
            vector_errors = (u_hat - u_all[i])[
                :, 0, :
            ]  # the indexing here is because OpenCV likes to add extra dimensions.
            scalar_errors = np.linalg.norm(vector_errors, axis=1)
            mean_errors.append(np.mean(scalar_errors))
        self.__save_txt(K, "K.txt")
        self.__save_txt(dc, "dc.txt")
        self.__save_txt(mean_errors, "mean_errors.txt")
        self.__save_txt(std_int, "std_int.txt")
        print(
            'Calibration data is saved in the folder "%s"'
            % realpath(self.output_folder)
        )

    def __calibrate(self):
        X_board = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        X_board[:, :2] = self.square_size * np.mgrid[
            0 : self.board_size[0], 0 : self.board_size[1]
        ].T.reshape(-1, 2)
        X_all = []
        u_all = []
        image_names = []
        image_size = None
        for image_path in glob.glob(self.image_path_pattern):
            print("%s..." % basename(image_path), end="")

            I = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
            if not image_size:
                image_size = I.shape
            elif I.shape != image_size:
                print("Image size is not identical for all images.")
                print(
                    'Check image "%s" against the other images.' % basename(image_path)
                )
                quit()

            ok, u = cv.findChessboardCorners(
                I, (self.board_size[0], self.board_size[1]), self.detect_flags
            )
            if ok:
                print("detected all %d checkerboard corners." % len(u))
                X_all.append(X_board)
                u = cv.cornerSubPix(I, u, (11, 11), (-1, -1), self.subpix_criteria)
                u_all.append(u)
                image_names.append(basename(image_path))
            else:
                print("failed to detect checkerboard corners, skipping.")

        with open(join(self.output_folder, "image_names.txt"), "w+") as f:
            for i, image_name in enumerate(image_names):
                f.write("%d: %s\n" % (i, image_name))

        return u_all, X_all, image_size

    def assess_uncertainty(self, N):

        image = join(self.output_folder, "checkerboard.png")
        camera_matrix = np.array(
            ((2359.40946, 0, 1370.05852), (0, 2359.61091, 1059.63818), (0, 0, 1))
        )
        size = (100, 10)  # width, height

        print(f"Generating {N} images with sampled distortion... ", end="")
        for i in range(N):

            # Parameters found from other script
            k1 = np.random.normal(-0.06652, 0.00109)
            k2 = np.random.normal(0.06534, 0.00624)
            k3 = np.random.normal(-0.07555, 0.01126)
            p1 = np.random.normal(0.00065, 0.00011)
            p2 = np.random.normal(-0.00419, 0.00014)

            distortion_coefficients = np.array((k1, k2, k3, p1, p2))
            new_camera_matrix, _ = cv.getOptimalNewCameraMatrix(
                camera_matrix, distortion_coefficients, size, 1, size
            )

            I = cv.imread(image, cv.IMREAD_GRAYSCALE)
            IU = np.empty_like(I)

            _ = cv.undistort(
                src=I,
                dst=IU,
                cameraMatrix=camera_matrix,
                distCoeffs=distortion_coefficients,
                newCameraMatrix=new_camera_matrix,
            )

            cv.imwrite(f"../data/distortion_sample/sample{i:03d}.png", IU)
        print("Done!")

    def run(self):

        if exists(join(self.output_folder, "u_all.npy")):
            print("Using previous checkerboard detection results.")
            u_all = self.__load_np("u_all.npy")
            X_all = self.__load_np("X_all.npy")
            image_size = self.__load_txt("image_size.txt")

        else:
            u_all, X_all, image_size = self.__calibrate()

            self.__save_txt(image_size, "image_size.txt")
            self.__save_np(u_all, "u_all.npy")
            self.__save_np(X_all, "X_all.npy")

        print("Calibrating...", end="")
        results = cv.calibrateCameraExtended(
            X_all, u_all, image_size, None, None, flags=self.calibrate_flags
        )
        print("Done!")

        self.save_calibration_results(results, X_all, u_all)


if __name__ == "__main__":
    calibrator = CameraCalibration()

    calibrator.run()

    calibrator.assess_uncertainty(10)
