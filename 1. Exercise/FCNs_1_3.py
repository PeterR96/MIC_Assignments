# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:45:04 2022

@author: Peter, Filip, Markus
"""

import math
import cv2
import numpy as np 
from matplotlib import pyplot as plt

def Sobel_kernel(img):
    
    lap = cv2.Laplacian(img,cv2.CV_32F) 
    sobelx = cv2.Sobel(img,ddepth=cv2.CV_32F,dx=1,dy=0)  # x
    sobely = cv2.Sobel(img,ddepth=cv2.CV_32F,dx=0,dy=1)  # y
    sobel = [sobelx,sobely]
    cv2.imwrite('Sobel_horizontal.tif', sobelx)
    cv2.imwrite('Sobel_vertical.tif', sobely)
    cv2.imwrite('Laplacian.tif', lap)
    
    return sobel

def Img_Gradient(sobel):
    
    mag = np.sqrt((sobel[0] ** 2) + (sobel[1] ** 2))
    orientation = np.arctan2(sobel[1], sobel[0]) * (180 / np.pi) % 180
    #cv2.imshow("gradient magnitude", mag)
    
    cv2.imwrite('Gradiant_Magnetut.tif', mag)
    cv2.imwrite('Orientation.tif', orientation)
    
    return mag
    return orientation