# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 08:59:23 2022

@author: Peter, Filip, Markus
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img_norm = cv2.imread("./Normalized_Img.tif", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
plt.figure(1)
plt.imshow(img_norm)

# Otsu's thresholding
img = np.uint8(img_norm*255)
plt.figure(2)
plt.imshow(img)
Otsu_th,Otsu_img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.figure(3)
plt.imshow(Otsu_img)
cv2.imwrite("Otsu_Mask.jpg", Otsu_img)


Triangle_th,Triangle_img = cv2.threshold(img,0,255,cv2.THRESH_TRIANGLE)
cv2.imwrite("Triangle_Mask.jpg", Triangle_img)


#img1 = cv2.imread("./Sobel_vertical.jpg", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
img1 = cv2.GaussianBlur(img, (11,11), 9)
Otsu_th1,Otsu_img1 = cv2.threshold(img1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite("Otsu_Mask_gauss.jpg", Otsu_img1)
Triangle_th1,Triangle_img1 = cv2.threshold(img1,0,255,cv2.THRESH_TRIANGLE)
cv2.imwrite("Triangle_Mask_gauss.jpg", Triangle_img1)
