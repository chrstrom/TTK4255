import numpy as np
from scipy import ndimage

def rgb_to_gray(I):
    """
    Converts a HxWx3 RGB image to a HxW grayscale image as
    described in the text.
    """
    R = I[:,:,0]
    G = I[:,:,1]
    B = I[:,:,2]

    #How not to do it, way too complicated:
    #N = np.shape(I)[0]
    #M = np.shape(I)[1]
#
    #Gray = np.zeros((N, M))
#
    #for i in range(N):
    #    for j in range(M):
    #        Gray[i, j] = np.average([R[i, j], G[i, j], B[i, j]])
    
    # How to do it, use smart syntax to write simple code:
    Gray = (R + G + B)/3
    return Gray # Placeholder

def central_difference(I):
    """
    Computes the gradient in the x and y direction using
    a central difference filter, and returns the resulting
    gradient images (Ix, Iy) and the gradient magnitude Im.
    """
    kernel = np.array([1/2, 0, -1/2])

    N = np.shape(I)[0]
    M = np.shape(I)[1]

    Ix = np.zeros_like(I) # Placeholder
    Iy = np.zeros_like(I) # Placeholder
    Im = np.zeros_like(I) # Placeholder

    for i in range(0,N):
        Ix[i, :] = np.convolve(I[i, :], kernel, "same")
    
    for j in range(0,M):
        Iy[:, j] = np.convolve(I[:, j], kernel, "same")
    
    # Another way of doing it :)
    #Ix = ndimage.convolve1d(I, kernel, 1)
    #Iy = ndimage.convolve1d(I, kernel, 0)
    Im = np.sqrt(Ix**2 + Iy**2)

    return Ix, Iy, Im

def gaussian(I, sigma):
    """
    Applies a 2-D Gaussian blur with standard deviation sigma to
    a grayscale image I.
    """

    # Hint: The size of the kernel should depend on sigma. A common
    # choice is to make the half-width be 3 standard deviations. The
    # total kernel width is then 2*np.ceil(3*sigma) + 1.
    
    K = round(2*np.pi*sigma) + 1

    N = np.shape(I)[0]
    M = np.shape(I)[1]

    ker_x = np.zeros(K)

    for i in range(K):
        bruh = (1/(2*np.pi*sigma**2)) * np.exp(- (i**2)/(2*sigma**2))
        ker_x[i] = bruh
    
    ker_y = ker_x
    
    result = np.zeros_like(I) # Placeholder
    
    for i in range(0,N):
        result[i, :] = np.convolve(I[i,:], ker_x, "same")
    for j in range(0,M):
        result[:, j] = np.convolve(I[:, j], ker_y, "same")
    return result

def extract_edges(Ix, Iy, Im, threshold):
    """
    Returns the x, y coordinates of pixels whose gradient
    magnitude is greater than the threshold. Also, returns
    the angle of the image gradient at each extracted edge.
    """
    x = []
    y = []
    theta = []

    print(Im)
    N = np.shape(Im)[0]
    M = np.shape(Im)[1]
    
    Im_thresh = Im[:,:] > threshold

    detection_indicies = np.nonzero(Im_thresh)

    for i in range(M):
        for j in range(N):
            if Im_thresh[j,i] == True:
                x.append(i)
                y.append(j)
                theta.append(np.arctan2(Iy[j, i], Ix[j, i]))
        
    #y = detection_indicies[0]
    #x = detection_indicies[1]
    #theta = np.arctan2(Iy[y, x], Ix[y, x])
    return x, y, theta # Placeholder
