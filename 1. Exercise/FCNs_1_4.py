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
def Otsu_Threshholding_Algorithm(img_norm):
    #compute threshold value and mask with otsu algorithm
    img = np.uint8(img_norm*255)               # convert float32 img into uint8
    Otsu_th,Otsu_img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite("Otsu_Mask.jpg", Otsu_img)
    return Otsu_th                             # return threshold value

def Triangle_Threshholding_Algorithm(img_norm):
    #compute threshold value and mask with triangle algorithm
    img = np.uint8(img_norm*255)               # convert float32 img into uint8
    Triangle_th,Triangle_img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
    cv2.imwrite("Triangle_Mask.jpg", Triangle_img)
    return Triangle_th                         # return threshold value
