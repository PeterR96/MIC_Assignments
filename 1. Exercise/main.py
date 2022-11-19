# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 21:00:23 2022

@author: Peter, Filip, Markus
"""
from FCNs_1_2 import *
from FCNs_1_3 import *
from FCNs_1_4 import *
import cv2

#import math
#import numpy as np 
#from matplotlib import pyplot as plt
#from PIL import Image

raw_img = cv2.imread("./OCTimage_raw.tif",cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

#variables
x, y = raw_img.shape
bits=2**16
n=5

PixelX=160  
PixelY=160

#treshholds
t_uint16=25000
t_uint8 = 220


"""2 OCT image preprocessing framework (find functions in "FCNs_1_2.py")----"""
#2.1 Histogram
display_Histogram(raw_img,bits)

#2.2 Image Transformations
log_transformed = Img_Log_Transformation(raw_img,bits)
gamma_transformed = Img_Gamma_Transformation(raw_img,bits)

#2.3 Spital Filter and Normalize filtered Img
AVG_Filtered_Img = Spital_Filter_AVG(log_transformed,n)
Gaussian_Filtered_Image = Spital_Filter_Gaussian(log_transformed,n)
Normalized_Img = Filtered_Img_Normalize(Gaussian_Filtered_Image,x,y)
display_norm_Hisogram (Normalized_Img)
#2.4 3 Pixel Neighborhood


Crop = Neighborhood(PixelX,PixelY,log_transformed)

#Neighborhood(Pixelx,Pixely,img)
#Neighborhood(Pixelx,Pixely,img)

"""
#2.5 Treshhold
#Threshold()
"""
"""3 Edge Detection (find functions in "FCNs_1_3.py")------------------------"""
#3.1 Edge Detection with Sobel kernel
Sobel_Filter = Sobel_kernel(Normalized_Img)

#3.2 Image Gradient and 3.3 Treshhold
Sobel_Gradient = Img_Gradient_Threshhold(Sobel_Filter,t_uint16)

#3.4 Canny Edge Detection
Canny_Edge = Canny_Edge_Detection(Normalized_Img,t_uint8)

"""4 Image Segmentation (find functions in "FCNs_1_4.py")--------------------"""
#4.1 Otsu Threshholding Algorithm
Otsu_threshold = Otsu_Threshholding_Algorithm(Normalized_Img)

#4.2 Additional Segmentation Algorithm
Triangle_Threshholding_Algorithm(Normalized_Img)

#4.3 Segment Evaluation

