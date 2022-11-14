# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 21:00:23 2022

@author: Peter Rott
"""
from FCNs_1_2 import *
from FCNs_1_3 import *
from FCNs_1_4 import *
import math
import cv2
import numpy as np 
from matplotlib import pyplot as plt
from PIL import Image

raw_img = cv2.imread("./OCTimage_raw.tif",cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

x, y = raw_img.shape
bits=2**16
n=5
"""2 OCT image preprocessing framework -------------------------------------"""
#2.1 Histogram
display_Histogram(raw_img,bits)

#2.2 Image Transformations
log_transformed = Img_Log_Transformation(raw_img,bits)
gamma_transformed = Img_Gamma_Transformation(raw_img,bits)

#2.3 Spital Filter and Normalize filtered Img
AVG_Filtered_Img = Spital_Filter_AVG(gamma_transformed,n)
Gaussian_Filtered_Image = Spital_Filter_Gaussian(gamma_transformed,n)
Normalized_Img = Filtered_Img_Normalize(Gaussian_Filtered_Image,x,y)

#<<<<<<< HEAD
""" 13.11.22 Markus commented because of compile error 
img_rgb = cv2.imread('raw_image', cv2.IMREAD_UNCHANGED)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
cv2.imwrite(img_rgb, img_rgb)
"""
#=======
 
"""img_rgb = cv2.imread('raw_image', cv2.IMREAD_UNCHANGED)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
cv2.imwrite(img_rgb, img_rgb)"""

#>>>>>>> 94153314f039c85a73aa56daed9994bb4d2033f2
# dont know what this is?
# hahaha 
    



"""
''''''#2.4 3 Pixel Neighborhood
gray = cv2.cvtColor('./OCTimage_raw.tif', cv2.COLOR_RGB2GRAY)
cv2.imshow(gray, gray_image)
cv2.waitKey(0) 

Neighborhood(Pixelx,Pixely,img)
Neighborhood(Pixelx,Pixely,img)
Neighborhood(Pixelx,Pixely,img)

#2.5 Treshhold
Threshold()
"""
#3-----------------------------------------------------------------------
#3.1 Edge Detection with Sobel kernel
Sobel_Filter = Sobel_kernel(Normalized_Img)

#3.2 Image Gradient
#Sobel_Array = np.array(Sobel_Filter)
G=Img_Gradient(Sobel_Filter)
"""
#3.3 Treshhold and most prominent boundries in the image
Threshold_boundries()

#3.4 Canny Edge Detection
Canny_Edge_Detection()
"""
#4-----------------------------------------------------------------------
#4.1 Otsu Threshholding Algorithm
#Otsu_threshold = Otsu_Threshholding_Algorithm(raw_img)

#4.2 Additional Segmentation Algorithm
#Adaptive_Gaussian_Threshholding_Algorithm(raw_img)

#4.3 Segment Evaluation
####""""------------------------------------------------------------------------"""
