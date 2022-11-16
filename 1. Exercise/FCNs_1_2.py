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
    plt.savefig('Histogram'+str(bits)+'.png', dpi=300, bbox_inches='tight')

def display_norm_Hisogram (img):
    histogram, bin_edges = np.histogram(img, bins=2*16, range=(0, 1))
    # configure and draw the histogram figure
    plt.figure()
    plt.title("Histogram")
    plt.xlabel("pixel value")
    plt.ylabel("count")  
    plt.plot(bin_edges[0:-1], histogram)
    plt.savefig('Histogram_norm.png', dpi=300, bbox_inches='tight')
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
    gamma = 0.1
      
    # Apply gamma correction. Save edited image.
    gamma_tranformed = np.array((bits-1)*(img / (bits-1)) ** gamma, dtype = 'uint16')
    cv2.imwrite('2.2_oct_gamma_transformed_'+str(gamma)+'.tif', gamma_tranformed) 
    return gamma_tranformed
    
"""2.3 Spital Filter"""
#Average Filter Gamma
def Spital_Filter_AVG(img,n):   
    kernel = np.ones((n,n),np.float32)/n**2
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

def Neighborhood(PixelX,PixelY,img):
    
    
    start_point = (PixelX-11, PixelY-11)
    # Ending coordinate
    # represents the bottom right corner of rectangle
    end_point = (PixelX+11 , PixelY+11)
    #  color in BGR
    color = (255, 255, 255)
    thickness = 1
    
    # Using cv2.rectangle() method
    # Draw a rectangle with blue line borders of thickness of 2 px
    selected_image = cv2.rectangle(img, start_point, end_point, color, thickness)
    # Displaying the image 
    cv2.imwrite('Area_of_interest.tif', selected_image)
    
    #standard_dev = np.std(selected_image)
    #mean1 = no.mean(selected_image)
    
   
   #img8 = np.uint8(selected_image*255)
  
    img_crop = img [(PixelX-10):(PixelY+11),(PixelY-10):(PixelY+11)] 
    cv2.imwrite('2.4_Area_of_interest_with_rectangle.tif', selected_image)
    cv2.imwrite('2.4_Area_of_interest_21x21.tif', img_crop)
    
   # img_crop = selected_image.crop(PixelX-10, PixelY+10, PixelX-10, PixelY+10)
    
    mean, std = cv2.meanStdDev(img_crop)
    
    
    print("Standard deviation of the selected area is: ", std)
    print("Mean value of the selected area is: ", mean)

    
   # return selected_image
    #return mean
    #return std
    return img_crop
