# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:45:04 2022

@author: Peter Rott
"""
"""2.4 Edge """
import math
import cv2
import numpy as np 
from matplotlib import pyplot as plt

def Sobel_kernel(img):

    sobelx = cv2.Sobel(img,cv2.CV_32F,1,0,ksize=5)  # x
    sobely = cv2.Sobel(img,cv2.CV_32F,0,1,ksize=5)  # y

    cv2.imwrite('Sobel_horizontal.jpg', sobelx)
    cv2.imwrite('Sobel_vertical.jpg', sobely)

"""laplacian_f = cv2.Laplacian(filtered_img,cv2.CV_64F)
sobelx_f = cv2.Sobel(filtered_img,cv2.CV_64F,1,0,ksize=5)  # x
sobely_f = cv2.Sobel(filtered_img,cv2.CV_64F,0,1,ksize=5)  # y

laplacian_n = cv2.Laplacian(norm_img,cv2.CV_64F)
sobelx_n = cv2.Sobel(norm_img,cv2.CV_64F,1,0,ksize=5)  # x
sobely_n = cv2.Sobel(norm_img,cv2.CV_64F,0,1,ksize=5)  # y

    
#print "Gradiant : ", math.sqrt(sobelx_f**2 + sobely_f**2)"""