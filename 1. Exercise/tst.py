import math
import cv2
import numpy as np 
from matplotlib import pyplot as plt

raw_img = cv2.imread("./Normalized_Img.tif",cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
#gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)                     
x, y = raw_img.shape
bits=2**16

gX = cv2.Sobel(raw_img, ddepth=cv2.CV_32F, dx=1, dy=0)
gY = cv2.Sobel(raw_img, ddepth=cv2.CV_32F, dx=0, dy=1)
mag = np.sqrt((gX ** 2) + (gY ** 2))
orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180

#cv2.imshow("gradient magnitude", mag)
cv2.imshow("gx", gX)
#cv2.imshow("by", gY)

cv2.waitKey(0)

#https://codereview.stackexchange.com/questions/192292/threshold-an-image-for-a-given-range-and-sobel-kernel
class GRADIENT_THRESHOLD(object):
    '''
    Define functions to threshold an image for a given range and Sobel kernel
    '''

    def __init__(self, args):
        self.args = args

    def gradient_abs_sobel(self, img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # take the absolute value of the gradient in given orient = 'x' or 'y'
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        # scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
        binary_output = np.zeros_like(scaled_sobel)
        # return this mask as your binary_output image
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        # return the binary image
        return binary_output 

    def gradient_magnitude(self, img, sobel_kernel=3, thresh=(0, 255)):
        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        # create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
        # return the binary image
        return binary_output