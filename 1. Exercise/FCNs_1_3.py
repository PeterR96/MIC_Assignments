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

def Sobel_kernel(img):
    
    lap = cv2.Laplacian(img,cv2.CV_32F,ksize=5) 
    lap = np.uint8(np.absolute(lap))

    sobelx= cv2.Sobel(img,cv2.CV_32F, dx=1,dy=0, ksize=5)
    #sobelx= np.uint8(np.absolute(sobelx))
 
    sobely= cv2.Sobel(img,cv2.CV_32F, dx=0,dy=1, ksize=5)
    #sobely = np.uint8(np.absolute(sobely))
    
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    
    sobel = [lap,sobelx,sobely]
    return sobel
"""
    sobelx = cv2.Sobel(img,cv2.CV_32F,1,0,ksize=5)  # x
    sobely = cv2.Sobel(img,cv2.CV_32F,0,1,ksize=5)  # y
    sobel = [sobelx,sobely]
    cv2.imwrite('Sobel_horizontal.jpg', sobelx)
    cv2.imwrite('Sobel_vertical.jpg', sobely)
    return sobel
"""
def Img_Gradient(sobel):
    
    
    """results = sobel
    images =["Gradient Img","Gradient_X","Gradient_Y"]
    plt.figure(figsize=(10,10))
    for i in range(3):
        plt.title(results[i])
        plt.subplot(1,3,i+1)
        plt.imshow(results[i],"plasma")
        plt.xticks([])
        plt.yticks([])
 
    plt.show()"""
"""laplacian_f = cv2.Laplacian(filtered_img,cv2.CV_64F)
sobelx_f = cv2.Sobel(filtered_img,cv2.CV_64F,1,0,ksize=5)  # x
sobely_f = cv2.Sobel(filtered_img,cv2.CV_64F,0,1,ksize=5)  # y

laplacian_n = cv2.Laplacian(norm_img,cv2.CV_64F)
sobelx_n = cv2.Sobel(norm_img,cv2.CV_64F,1,0,ksize=5)  # x
sobely_n = cv2.Sobel(norm_img,cv2.CV_64F,0,1,ksize=5)  # y

    
#print "Gradiant : ", math.sqrt(sobelx_f**2 + sobely_f**2)"""