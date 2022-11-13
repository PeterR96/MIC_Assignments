# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 16:45:54 2022

@author: Markus
"""

import cv2
import numpy as np 
from matplotlib import pyplot as plt

bits=2**16

"""2 OCT image preprocessing framework"""

"""Histogram"""
def display_Histogram(img,bits):
      
    histogram, bin_edges = np.histogram(img, bins=bits, range=(0, bits))
    # configure and draw the histogram figure
    plt.figure()
    plt.title("Histogram")
    plt.xlabel("pixel value")
    plt.ylabel("count")  
    plt.plot(bin_edges[0:-1], histogram)
    plt.savefig('Histogram.png', dpi=300, bbox_inches='tight')

"""2.2 Intensity transformation"""

"""Log Transfomation"""
def Img_Log_Transformation(img,bits):
    c = (bits-1)/(np.log(1 + np.max(img)))
    log_transformed = c * np.log(1 + img)
  
    # Specify the data type.Save the output.
    log_transformed = np.array(log_transformed, dtype = np.uint16)
    cv2.imwrite('oct_log_transformed.tif', log_transformed)
    return log_transformed

"""Gamma Transformation"""
def Img_Gamma_Transformation(img,bits):
    gamma = 0.2
      
    # Apply gamma correction. Save edited image.
    gamma_tranformed = np.array((bits-1)*(img / (bits-1)) ** gamma, dtype = 'uint16')
    cv2.imwrite('oct_gamma_transformed_'+str(gamma)+'.tif', gamma_tranformed) 
    return gamma_tranformed
    

"""2.3 Spital Filter"""
#Average Filter Gamma
def Spital_Filter_AVG(img,n):   
    kernel = np.ones((n,n),np.float32)/n**2
    AVG_Filtered_Img = cv2.filter2D(img,-1,kernel)
    cv2.imwrite('AVG_Filtered_Img'+str(n)+'x'+str(n)+'.tif', AVG_Filtered_Img)
    return AVG_Filtered_Img

#Gaussian Filter Gamma
def Spital_Filter_Gaussian(img,n):
    Gaussian_Filtered_Img = cv2.GaussianBlur(img, (n,n), 0)
    cv2.imwrite('Gaussian_Filtered_Img'+str(n)+'x'+str(n)+'.tif', Gaussian_Filtered_Img)
    return Gaussian_Filtered_Img

""" Normalize the Image"""
def Filtered_Img_Normalize(img,x,y):
    norm_img = np.zeros((x,y))
    final_img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    #cv2.imshow('Normalized_Img', final_img)
    cv2.imwrite('Normalized_Img.jpg', final_img)
    return final_img

"""2.4"""
"""
#21x21 neighborhood
 #import image
in_image=cv2.imread('oct_norm_gaussian.tif')
#convert the image to grayscale
#gray= cv2.cvtColor('oct_norm_gaussian.tif', cv2.COLOR_BGR2GRAY())
#select the pixel
pix=in_image[500][500]

#make a triangle from that pixel
#cv2.rectangle(gray, pt1, pt2, red)
#(means, stds) = cv2.meanStdDev(pix)
#wmean,bmean=means.flatten()
#wstds,bstds=stds.flatten()
#print ("means %.1f %.1f %.1f " % (wmean,bmean))
#print ("stds  %.1f %.1f %.1f " % (wstds,bstds))



""""""Plotting

titles=['Original','oct_gamma_transformed','Averaging Filter Gamma','Gaussian Filter Gamma',
        'Averaging Filter Log','Gaussian Filter Log','oct_norm_avg','oct_norm_gaussian']
images=[oct_image,gamma_corrected,dst,blur,dst1,blur1,dst2,blur2]

plt.figure('images')
for i in range(8):
    plt.subplot(4,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

# create the histograms
for j in range(8):
    plt.subplot(4,2,j+1)
    histogram, bin_edges = np.histogram(images[j], bins=bits, range=(0, bits))
    plt.title(titles[j])
    plt.xlabel("pixel value")
    plt.ylabel("count")
    histplot = plt.plot(bin_edges[0:-1], histogram)
"""
    
