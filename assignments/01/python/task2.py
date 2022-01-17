#!/usr/bin/python3

from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
from common import *

def rgb_split_channels(image_in):
    """
    Task 2.2
    """
    fig, ax = plt.subplots(1, 4, sharey='row')
    ax[0].imshow(image_in)
    ax[0].set_title("Normal image")

    ax[1].imshow(image_in[:,:,0])
    ax[1].set_title("Channel 0")

    ax[2].imshow(image_in[:,:,1])
    ax[2].set_title("Channel 1")

    ax[3].imshow(image_in[:,:,2])
    ax[3].set_title("Channel 2")

    plt.tight_layout()
    plt.show()

    return 0

def rgb_naive_threshold_on_green(image_in, threshold):
    """
    Task 2.3
    """
    image_green_threshold = image_in[:,:,1] > threshold

    fig, ax = plt.subplots(1, 2, sharey='row')
    plt.set_cmap('gray')
    ax[0].imshow(image_in)
    ax[0].set_title("Normal image")

    ax[1].imshow(image_green_threshold)
    ax[1].set_title(f"Thresholded on green > {threshold}")

    plt.tight_layout()
    plt.show()

    return 0


def rgb_normalized_values(image_in):
    """
    Task 2.4
    """
    R = np.array(image_in[:,:,0])
    G = np.array(image_in[:,:,1])
    B = np.array(image_in[:,:,2])

    color_channel_sum = R + G + B
    epsilon = 1e-10 # Avoid div by 0

    image_out = np.zeros(image_in.shape)
    image_out[:,:,0] = R / (color_channel_sum + epsilon)
    image_out[:,:,1] = G / (color_channel_sum + epsilon)
    image_out[:,:,2] = B / (color_channel_sum + epsilon)

    return image_out

def rgb_normalized_threshold_on_green(image_in, threshold):
    """
    Task 2.5
    """
    image_green_threshold = rgb_normalized_values(image_rgb)
    image_green_threshold = image_green_threshold[:,:,1] > threshold

    fig, ax = plt.subplots(1, 2, sharey='row')
    plt.set_cmap('gray')

    ax[0].imshow(image_in)
    ax[0].set_title("Normal image")

    ax[1].imshow(image_green_threshold)
    ax[1].set_title(f"Thresholded on green > {threshold}")

    plt.tight_layout()
    plt.show()

    return 0

if __name__ == '__main__':
    threshold = 0.45
    sigma     = 1 # todo: choose an appropriate value
    filename  = '../data/grass.jpg'

    image_rgb = plt.imread(filename)
    image_rgb = image_rgb/255.0

    rgb_split_channels(image_rgb)

    rgb_naive_threshold_on_green(image_rgb, threshold)

    rgb_normalized_threshold_on_green(image_rgb, threshold)
