# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 16:04:33 2022

@author: Peter, Filip, Markus
"""

import cv2
import numpy as np 

""" 4. Image Segmentation """

# 4.1 Otsu thresholding algorithm
def Otsu_Threshholding_Algorithm(img_norm):
    #compute threshold value and mask with otsu algorithm
    img = np.uint16(img_norm*255)
    Otsu_th,Otsu_img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite("4.1_Otsu_Mask.jpg", Otsu_img)
    return Otsu_th                    # return threshold value

# 4.1 Triangle threshholding
def Triangle_Threshholding_Algorithm(img_norm):
    img = np.uint8(img_norm*255)
    Triangle_th,Triangle_img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
    cv2.imwrite("4.2_Triangle_Mask.jpg", Triangle_img)
    return Triangle_th                 # return threshold value
