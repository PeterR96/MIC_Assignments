# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:45:04 2022

@author: Peter, Filip, Markus
"""

import cv2
import numpy as np 

def Sobel_kernel(img):
    
    sobelx = cv2.Sobel(img,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=3)  # x
    sobely = cv2.Sobel(img,ddepth=cv2.CV_32F,dx=0,dy=1,ksize=3)  # y
    sobel = [sobelx,sobely]
    cv2.imwrite('3.1_Sobel_horizontal.tif', sobelx)
    cv2.imwrite('3.1_Sobel_vertical.tif', sobely)
    return sobel

def Img_Gradient_Threshhold(sobel,thresh):
    #Gradient
    mag = np.sqrt((sobel[0] ** 2) + (sobel[1] ** 2))
    scale_factor = np.max(mag)/65535
    mag = (mag/scale_factor).astype(np.uint16) 
 
    #Threshhold
    ret,threshhold = cv2.threshold(mag,thresh,65535,cv2.THRESH_BINARY)
    
    #save images
    cv2.imwrite('3.2_Gradiant_Magnetut.tif', mag)
    cv2.imwrite('3.3_Gradiant_Magnetut_Threshhold_'+str(thresh)+'.tif', threshhold)
    return ret

def Canny_Edge_Detection(img,thresh):
    # Applying the Canny Edge filter
    img8 = np.uint8(img*255)
    c_edge = cv2.Canny(img8,thresh,255)
    cv2.imwrite('3.4_Canny_Edge_Detection.tif', c_edge)
   
    return c_edge