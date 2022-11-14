# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 10:37:43 2022

@author: filip
"""

import math
import cv2
import numpy as np 
from matplotlib import pyplot as plt
from PIL import Image


#21x21 neighborhood
 #import image
image1= cv2.imread('./oct_log_transformed.tif', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
#image is grayscale already

#select the pixel
# Start coordinate
# represents the top left corner of rectangle
start_point = (50, 50)
  
# Ending coordinate
# represents the bottom right corner of rectangle
end_point = (60, 61)
  
# Blue color in BGR
color = (0, 255, 0)
  
# Line thickness of 2 px
thickness = 1
  
# Using cv2.rectangle() method
# Draw a rectangle with blue line borders of thickness of 2 px
image = cv2.rectangle(image1, start_point, end_point, color, thickness)
cv2.imshow('image', image)
cv2.waitKey(0) 
  
#closing all open windows 
cv2.destroyAllWindows() 
# Displaying the image 
cv2.imwrite('Area_of_interest.tif', image)