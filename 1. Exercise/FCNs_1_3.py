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
    
    sobelx = cv2.Sobel(img,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=3)  # x
    sobely =  cv2.Sobel(img,ddepth=cv2.CV_32F,dx=0,dy=1,ksize=3)  # y
    sobel = [sobelx,sobely]
    cv2.imwrite('Sobel_horizontal.tif', sobelx)
    cv2.imwrite('Sobel_vertical.tif', sobely)
 
    
    return sobel

def Img_Gradient(sobel,thresh):
    
    mag = np.sqrt((sobel[0] ** 2) + (sobel[1] ** 2))
    
    scale_factor = np.max(mag)/65535
    mag = (mag/scale_factor).astype(np.uint16) 
 
    #Threshhold
    ret,thresh2 = cv2.threshold(mag,thresh,65535,cv2.THRESH_BINARY)
    cv2.imwrite('Gradiant_Magnetut.tif', mag)
    cv2.imwrite('Gradiant_Magnetut_Threshhold.tif', thresh2)
    #return binary_output
    return thresh2
