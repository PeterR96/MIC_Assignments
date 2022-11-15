# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 09:14:08 2022

@author: Peter Rott
"""

import cv2
import numpy as np 

bits=2**16
t_low=125
t_up = 255

img_norm = cv2.imread("./Normalized_Img.tif",cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

img = np.uint8(img_norm*255)
c_edge = cv2.Canny(img,t_low,t_up)
cv2.imwrite('Canny_Edge_Detection.tif', c_edge)

