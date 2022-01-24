
from ctypes import sizeof
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image

########################## 2.1) ###############################

filename_grass = '../data/grass.jpg'

Img = Image.open(filename_grass)

Img_rgb = np.array(Img) / 255.0

#print(np.shape(Img_rgb)) 

########################## 2.2) ###############################
Img_rgb0 = Img_rgb[:,:,0]
Img_rgb1 = Img_rgb[:,:,1]
Img_rgb2 = Img_rgb[:,:,2]

#plt.imshow(Img_rgb0, "gray", vmin = 0, vmax = 255)
#
#plt.imshow(Img_rgb1, "gray", vmin = 0, vmax = 255)
#plt.show()
#
#plt.imshow(Img_rgb2, "gray", vmin = 0, vmax = 255)
#plt.show()

def plot_img(Img):

    plt.imshow(Img, cmap="gray", vmin= 0, vmax= 1)
    plt.show()

def plot_img_channels(Img_rgb):
    16.01
    plt.figure(figsize=(14,3))

    plt.subplot(141)

    plt.imshow(Img_rgb, cmap = "gray")

    plt.title('RGB')

    plt.subplot(142)

    plt.imshow(Img_rgb[:,:,0], cmap='gray', vmin = 0, vmax = 1)

    plt.title('r')

    plt.subplot(143)

    plt.imshow(Img_rgb[:,:,1], cmap='gray', vmin = 0, vmax = 1)

    plt.title('g')

    plt.subplot(144)

    plt.imshow(Img_rgb[:,:,2], 'gray', vmin = 0, vmax = 1)

    plt.title('b')

    plt.tight_layout()

    # plt.savefig('out_normalized_rgb.pdf') # Uncomment to save figure in working directory

    plt.show()

#plot_img_channels(Img_rgb)
##############################################################

######################### 2.3) ###############################

def thresholding(Img, k):

    shape = np.shape(Img)
    N = shape[0]
    M = shape[1]

    for i in range(N):
        for j in range(M):
            if Img[i, j] > k:
                Img[i, j] = 1.0
            else:
                Img[i, j] = 0.0
    return Img

Img_thresheld = thresholding(Img_rgb1, 0.7)
#
#mask = Img_rgb[:,:,1] > 0.4 #value
#
#print(mask)
#print(np.max(mask))
#
#plt.imsave('output.jpg', mask, cmap='gray')

#plt.imshow(Img_thresheld, cmap = "gray")
#plt.show()


###################### 2.4) ##################################

def normalize_rbg(Img):

    shape = np.shape(Img)
    N = shape[0]
    M = shape[1]
    eps = 0.0001

    R = Img[:,:,0]
    G = Img[:,:,1]
    B = Img[:,:,2]
    
    sums = R + G + B

    result = np.zeros_like(Img)

    result[:,:,0] = R / (sums + eps)
    result[:,:,1] = G / (sums + eps)
    result[:,:,2] = B / (sums + eps)
    
    return result


Img_rgb_norm = normalize_rbg(Img_rgb)


#print(Img_rgb_norm)
#print(np.shape(Img_rgb_norm))

plot_img_channels(Img_rgb_norm)

threshhold_normalized = thresholding(Img_rgb_norm[:, :, 1], 0.4)

plot_img(threshhold_normalized)

