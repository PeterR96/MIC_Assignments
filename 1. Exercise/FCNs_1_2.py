# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 16:45:54 2022

@author: Peter, Filip, Markus
"""

import cv2
import numpy as np 
from matplotlib import pyplot as plt
from PIL import Image

"""2 OCT image preprocessing framework"""
"""Histogram"""
def display_Histogram(img,bits):
      
    histogram, bin_edges = np.histogram(img, bins=bits, range=(0, 255))
    # configure and draw the histogram figure
    plt.figure()
    plt.title("Histogram")
    plt.xlabel("pixel value")
    plt.ylabel("count")  
    plt.plot(bin_edges[0:-1], histogram)
    plt.savefig('2.1_Histogram_raw_OTC.png', dpi=300, bbox_inches='tight')

def display_norm_Hisogram (img):
    histogram, bin_edges = np.histogram(img, bins=2**16, range=(0, 1))
    # configure and draw the histogram figure
    plt.figure()
    plt.title("Histogram normalized Image")
    plt.xlabel("pixel value")
    plt.ylabel("count")  
    plt.plot(bin_edges[0:-1], histogram)
    plt.savefig('2.3_Histogram_norm.png', dpi=300, bbox_inches='tight')
"""2.2 Intensity transformation"""
"""Log Transfomation"""

def Img_Log_Transformation(img,bits):
    c = (bits-1)/(np.log(1 + np.max(img)))
    log_transformed = c * np.log(1 + img)
  
    # Specify the data type.Save the output.
    log_transformed = np.array(log_transformed, dtype = np.uint16)
    cv2.imwrite('2.2_oct_log_transformed.tif', log_transformed)
    return log_transformed

"""Gamma Transformation"""
def Img_Gamma_Transformation(img,bits):
    gamma = 0.2
      
    # Apply gamma correction. Save edited image.
    gamma_tranformed = np.array((bits-1)*(img / (bits-1)) ** gamma, dtype = 'uint16')
    cv2.imwrite('2.2_oct_gamma_transformed_'+str(gamma)+'.tif', gamma_tranformed) 
    return gamma_tranformed
    
"""2.3 Spital Filter"""
#Average Filter Gamma
def Spital_Filter_AVG(img,n):   
    kernel = np.ones((n,n),np.uint16)/n**2
    AVG_Filtered_Img = cv2.filter2D(img,-1,kernel)
    cv2.imwrite('2.3_AVG_Filtered_Img'+str(n)+'x'+str(n)+'.tif', AVG_Filtered_Img)
    return AVG_Filtered_Img

#Gaussian Filter Gamma
def Spital_Filter_Gaussian(img,n):
    Gaussian_Filtered_Img = cv2.GaussianBlur(img, (n,n), 0)
    cv2.imwrite('2.3_Gaussian_Filtered_Img'+str(n)+'x'+str(n)+'.tif', Gaussian_Filtered_Img)
    return Gaussian_Filtered_Img

""" Normalize the Image"""
def Filtered_Img_Normalize(img,x,y):
    
    norm_img = np.zeros((x,y))
    final_img = cv2.normalize(img, norm_img, 0, 1, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    cv2.imwrite('2.3_Normalized_Img.tif', final_img)
    return final_img

"""2.4"""

def Neighborhood(PixelX,PixelY,PixelX1,PixelY1,PixelX2,PixelY2,img):
    
    #represent the points of rectangle
    start_point = (PixelX-11, PixelY-11)
    end_point = (PixelX+11 , PixelY+11)
    
    
    color = (255, 255, 255)
    thickness = 1

    
    # Draw a rectangle with blue line borders of thickness
    selected_image = cv2.rectangle(img, start_point, end_point, color, thickness)
    # Displaying the image 
    cv2.imwrite('Area_of_interest.tif', selected_image)
    img_crop = img [(PixelX-10):(PixelY+11),(PixelY-10):(PixelY+11)] 
    img_crop1 = img [(PixelX1-10):(PixelY1+11),(PixelY1-10):(PixelY1+11)] 
    img_crop2 = img [(PixelX2-10):(PixelY2+11),(PixelY2-10):(PixelY2+11)] 
    
    cv2.imwrite('2.4_Area_of_interest_with_rectangle.tif', selected_image)
    cv2.imwrite('2.4_Area_of_interest_21x21.tif', img_crop)
    cv2.imwrite('2.4_Area_of_interest_21x21a1.tif', img_crop1)
    cv2.imwrite('2.4_Area_of_interest_21x21a2.tif', img_crop2)
    
    #calculate the Standard deviation
    mean, std = cv2.meanStdDev(img_crop2)
    
    
    print("Standard deviation of the selected area is: ", std)
    print("Mean value of the selected area is: ", mean)

    histogram, bin_edges = np.histogram(img_crop2, bins=2**10, range=(0, 0.56))
      # configure and draw the histogram figure
    plt.figure()
    plt.title("Histogram crop 21x21 Pixel: "+str(PixelX2)+":"+str(PixelY2))
    plt.xlabel("pixel value")
    plt.ylabel("count")  
    plt.plot(bin_edges[0:-1], histogram)
    plt.savefig('2.4_Histogram_crop_21x21_norm_2_12.png', dpi=300, bbox_inches='tight')
    
   
    return img_crop

def Threshold_diff(t,t1,raw_img):
    
  img = np.uint16(raw_img*255)
  ret, thresh_hold2 = cv2.threshold(img,t,t1,  cv2.THRESH_BINARY)
  thresh_hold2 = cv2.resize(thresh_hold2, (960, 540))    
  cv2.imwrite ('Threshold_diff.tif', thresh_hold2) 
  
  return thresh_hold2