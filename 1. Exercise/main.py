# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 21:00:23 2022

@author: Peter Rott
"""
from FCNs_1_2 import *
import math
import cv2
import numpy as np 
from matplotlib import pyplot as plt

raw_img = cv2.imread("./OCTimage_raw.tif", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
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
AVG_Filtered_Img = Spital_Filter_AVG(log_transformed,n)
Gaussian_Filtered_Image = Spital_Filter_Gaussian(log_transformed,n)
Normalized_Img = Filtered_Img_Normalize(Gaussian_Filtered_Image,x,y)

"""
#2.4 3 Pixel Neighborhood
Neighborhood(Pixelx,Pixely,img)
Neighborhood(Pixelx,Pixely,img)
Neighborhood(Pixelx,Pixely,img)

#2.5 Treshhold
Threshold()

#3-----------------------------------------------------------------------
#3.1 Edge Detection with Sobel kernel
Sobel_kernel()

#3.2 Image Gradient
Img_Gradient()

#3.3 Treshhold and most prominent boundries in the image
Threshold_boundries()

#3.4 Canny Edge Detection
Canny_Edge_Detection()

#4-----------------------------------------------------------------------
#4.1 Otsu Threshholding Algorithm
Otsu_Threshholding_Algorithm()

#4.2 Additional Segmentation Algorithm
Additional_Segmentation_Algorithm()

#4.3 Segment Evaluation
#""""------------------------------------------------------------------------"""
