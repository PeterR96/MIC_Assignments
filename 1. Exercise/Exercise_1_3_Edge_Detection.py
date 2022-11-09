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

oct_image = cv2.imread("./OCTimage_raw.tif", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
filtered_img = cv2.imread("./oct_gamma_transformed0.2.tif", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
norm_img = cv2.imread("./otc_normalized.jpg", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
laplacian = cv2.Laplacian(oct_image,cv2.CV_64F)
sobelx = cv2.Sobel(oct_image,cv2.CV_64F,1,0,ksize=5)  # x
sobely = cv2.Sobel(oct_image,cv2.CV_64F,0,1,ksize=5)  # y

laplacian_f = cv2.Laplacian(filtered_img,cv2.CV_64F)
sobelx_f = cv2.Sobel(filtered_img,cv2.CV_64F,1,0,ksize=5)  # x
sobely_f = cv2.Sobel(filtered_img,cv2.CV_64F,0,1,ksize=5)  # y

laplacian_n = cv2.Laplacian(norm_img,cv2.CV_64F)
sobelx_n = cv2.Sobel(norm_img,cv2.CV_64F,1,0,ksize=5)  # x
sobely_n = cv2.Sobel(norm_img,cv2.CV_64F,0,1,ksize=5)  # y



image = [laplacian, laplacian_f,laplacian_n, sobelx, sobelx_f,sobelx_n, sobely, sobely_f, sobely_n]
title = ["laplacian_org","laplacian_filtered", "laplacian_norm","sobelx_org","sobelx_filtered","sobelx_norm","sobely_org","sobely_filtered","sobely_norm"]

plt.figure('images')
for i in range(9):
    plt.subplot(3,3,i+1),plt.imshow(image[i],'gray')
    plt.title(title[i])
    plt.xticks([]), plt.yticks([])
    
    
#print "Gradiant : ", math.sqrt(sobelx_f**2 + sobely_f**2)