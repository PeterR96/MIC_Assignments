# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 16:04:33 2022

@author: Peter, Filip, Markus
"""

import cv2
import numpy as np 
from matplotlib import pyplot as plt

""" 4. Image Segmentation """

# 4.1 Otsu thresholding algorithm
def Otsu_Threshholding_Algorithm(img):
    #compute threshold value and mask with otsu algorithm
    Otsu_th,Otsu_img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite("Otsu_Mask.jpg", Otsu_img)
    return Otsu_th                     # return threshold value
"""
def Adaptive_Gaussian_Threshholding_Algorithm(img):
    AGT_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    plt.figure()
    plt.title("Mask generated from Adaptive Gaussian Thresholding Algorithm using Otsu")
    plt.imshow(AGT_img,'gray')
    plt.savefig('AGT_Mask.png', dpi=600, bbox_inches='tight')
"""