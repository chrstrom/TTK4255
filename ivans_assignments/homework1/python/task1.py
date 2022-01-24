import numpy as np
import matplotlib.pyplot as plt
from common import *
from scipy.signal import convolve2d

filename  = '../data/xetrov.jpg'

I_rgb      = plt.imread(filename)
I_rgb      = I_rgb/255.0
I_gray     = rgb_to_gray(I_rgb)


def kursed_kernel(I):
    kernel = np.array([[-0.5, -1.0, -0.5], [-1.0, 7.0, -1.0], [-0.5, -1.0, -0.5]])

    N = np.shape(I)[0]
    M = np.shape(I)[1]

    
    I[:, 0:(M//2)] = convolve2d(I[:, 0:(M//2)], kernel, mode="same")
    print(np.shape(I))
    #I_cursed = np.array([I_2[:, :], I[:, 203:400]])
    return I


I_cursed = kursed_kernel(I_gray)

plt.imshow(I_cursed, cmap="gray")
plt.show()