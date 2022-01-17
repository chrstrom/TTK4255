import numpy as np
from scipy import ndimage


def rgb_to_gray(I):
    """
    Converts a HxWx3 RGB image to a HxW grayscale image as
    described in the text.
    """
    R = I[:, :, 0]
    G = I[:, :, 1]
    B = I[:, :, 2]

    return (R + G + B) / 3.0


def central_difference(I):
    """
    Computes the gradient in the x and y direction using
    a central difference filter, and returns the resulting
    gradient images (Ix, Iy) and the gradient magnitude Im.
    """
    kernel = np.array([0.5, 0, -0.5])
    Ix = ndimage.convolve1d(I, kernel, axis=1)
    Iy = ndimage.convolve1d(I, kernel, axis=0)
    # TODO: Implement without convolution functions as an exercise
    Im = np.sqrt(Ix ** 2 + Iy ** 2)
    return Ix, Iy, Im


def gaussian(I, sigma):
    """
    Applies a 2-D Gaussian blur with standard deviation sigma to
    a grayscale image I.
    """
    kernel_width = int(2 * np.ceil(3 * sigma) + 1)

    kernel = [
        1 / (2 * np.pi * sigma ** 2) * np.exp(-(x ** 2) / (2 * sigma ** 2))
        for x in range(-kernel_width // 2, kernel_width // 2)
    ]

    # TODO: Implement without convolution functions as an exercise
    horizontal_pass = ndimage.convolve1d(I, kernel, axis=1)
    vertical_pass = ndimage.convolve1d(horizontal_pass, kernel, axis=0)
    return vertical_pass


def extract_edges(Ix, Iy, Im, threshold):
    """
    Returns the x, y coordinates of pixels whose gradient
    magnitude is greater than the threshold. Also, returns
    the angle of the image gradient at each extracted edge.
    """
    magnitude_image_filtered = Im > threshold
    detection_indicies = np.nonzero(magnitude_image_filtered)

    y = detection_indicies[0]
    x = detection_indicies[1]
    theta = np.arctan2(Iy[y, x], Ix[y, x])

    return x, y, theta
