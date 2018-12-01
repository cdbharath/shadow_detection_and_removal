import numpy as np
import cv2

# cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
# ksize - size of gabor filter (n, n)
# sigma - standard deviation of the gaussian function
# theta - orientation of the normal to the parallel stripes
# lambda - wavelength of the sunusoidal factor
# gamma - spatial aspect ratio
# psi - phase offset
# ktype - type and range of values that each pixel in the gabor kernel can hold

img = cv2.imread('shadow.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

feature_vector = []
kernel_vector = []
scales = 4
orientation = 6

for s in range(scales):
    for o in range(orientation):
        g_kernel = cv2.getGaborKernel((21, 21), (4.0+s*2), (np.pi*o)/6, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
        kernel_vector.append(g_kernel)
        feature_vector.append(filtered_img)

print(filtered_img.shape)
cv2.imshow('image', img)
cv2.imshow('filtered image', filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
