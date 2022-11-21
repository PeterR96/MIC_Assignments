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

raw_img = cv2.imread("./OCTimage_raw.tif",cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
img = np.uint16(raw_img*255)

#ret, thresh_hold = cv2.threshold(img,  255, 1000, cv2.THRESH_BINARY) 
#ret, thresh_hold1 = cv2.threshold(img,  255, 550,cv2.THRESH_BINARY_INV) 
ret, thresh_hold2 = cv2.threshold(img, 3728,65535,  cv2.THRESH_BINARY) 
#ret, thresh_hold3 = cv2.threshold(img, 255, 550, cv2.THRESH_TOZERO_INV) 
#ret, thresh_hold4 = cv2.threshold(img,  255, 550, cv2.THRESH_TRUNC)   
 
#thresh_hold = cv2.resize(thresh_hold, (960, 540))    
#cv2.imshow('Binary Threshold Image', thresh_hold) 
 
#thresh_hold1 = cv2.resize(thresh_hold1, (960, 540))    
#cv2.imshow('Binary Threshold Inverted Image', thresh_hold1) 
 
thresh_hold2 = cv2.resize(thresh_hold2, (960, 540))    
cv2.imwrite ('Threshold Tozero Image.tif', thresh_hold2) 
 
#thresh_hold3 = cv2.resize(thresh_hold3, (960, 540))    
#cv2.imshow('ThresholdTozero Inverted output', thresh_hold3) 
 
#thresh_hold4= cv2.resize(thresh_hold4, (960, 540))    
#cv2.imshow('Truncated Threshold output', thresh_hold4) 
 
#if cv2.waitKey(0):  
 #   cv2.destroyAllWindows()
    
    
#ret, thresh_hold = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
#thresh_hold = cv2.resize(thresh_hold, (260, 140))    
#cv2.imwrite('2.4_Binary_Threshold_Image', thresh_hold)
histogram, bin_edges = np.histogram(img_crop1, bins=2**16, range=(0, 1))
  # configure and draw the histogram figure for the 2nd part
plt.figure()
plt.title("Histogram crop 21x21 Pixel: "+str(PixelX1)+":"+str(PixelY1))
plt.xlabel("pixel value")
plt.ylabel("count")  
plt.plot(bin_edges[0:-1], histogram)
plt.savefig('2.4_Histogram_crop_21x21.png', dpi=300, bbox_inches='tight')

histogram, bin_edges = np.histogram(img_crop2, bins=2**16, range=(0, 1))
  # configure and draw the histogram figure
plt.figure()
plt.title("Histogram crop 21x21 Pixel: "+str(PixelX2)+":"+str(PixelY2))
plt.xlabel("pixel value")
plt.ylabel("count")  
plt.plot(bin_edges[0:-1], histogram)
plt.savefig('2.4_Histogram_crop_21x21.png', dpi=300, bbox_inches='tight')

start_point1 = (PixelX1-11, PixelY1-11)
end_point1 = (PixelX1+11, PixelY1+11)

start_point2 = (PixelX2-11, PixelY2-11)
end_point2 = (PixelX2+11 , PixelY2+11)

 histogram, bin_edges = np.histogram(img_crop1, bins=2**16, range=(0, 1))
   # configure and draw the histogram figure for the 2nd crop
 plt.figure()
 plt.title("Histogram crop 21x21 Pixel: "+str(PixelX1)+":"+str(PixelY1))
 plt.xlabel("pixel value")
 plt.ylabel("count")  
 plt.plot(bin_edges[0:-1], histogram)
 plt.savefig('2.4_Histogram_crop_21x21a1.png', dpi=300, bbox_inches='tight')

 histogram, bin_edges = np.histogram(img_crop2, bins=2**16, range=(0, 1))
   # configure and draw the histogram figure for 3rd crop
 plt.figure()
 plt.title("Histogram crop 21x21 Pixel: "+str(PixelX2)+":"+str(PixelY2))
 plt.xlabel("pixel value")
 plt.ylabel("count")  
 plt.plot(bin_edges[0:-1], histogram)
 plt.savefig('2.4_Histogram_crop_21x21a2.png', dpi=300, bbox_inches='tight')
