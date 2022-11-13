# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 08:59:23 2022

@author: Peter, Filip, Markus
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("./OCTimage_raw.tif", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
#plt.imshow(img)

# Otsu's thresholding
Otsu_th,Otsu_img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite("Otsu_Mask.jpg", Otsu_img)


img = np.uint8(img)
Triangle_th,Triangle_img = cv2.threshold(img,0,255,cv2.THRESH_TRIANGLE)
cv2.imwrite("Triangle_Mask.jpg", Triangle_img)

img1 = cv2.imread("./Sobel_vertical.jpg", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
img1 = cv2.GaussianBlur(img1, (55,55), 10)
Otsu_th1,Otsu_img1 = cv2.threshold(img1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite("Otsu_Mask1.jpg", Otsu_img1)