# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 16:45:54 2022

@author: Peter, Filip, Markus
"""

import cv2
import numpy as np 
from matplotlib import pyplot as plt


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
    final_img = cv2.normalize(img, norm_img, 0, 1, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    #cv2.imshow('Normalized_Img', final_img)
    cv2.imwrite('Normalized_Img.tif', final_img)
    #cv2.waitKey(0)
    return final_img

"""2.4"""

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
  
#  color in BGR
color = (255, 255, 255)
  
# Line thickness of 2 px
thickness = 1
  
# Using cv2.rectangle() method
# Draw a rectangle with blue line borders of thickness of 2 px
image = cv2.rectangle(image1, start_point, end_point, color, thickness)
window_name='image'
# Displaying the image 
cv2.imwrite('Area_of_interest.tif', image)
#crop the image and save it
crop= image [50:72, 100:122]
#img2 = np.copy(image1)
#image1[50:72, 100:122] = 255

cv2.imshow('original', image1)
cv2.imshow('cropped', crop)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite('cropped', crop)

#(means, stds) = cv2.meanStdDev(pix)
#wmean,bmean=means.flatten()
#wstds,bstds=stds.flatten()
#print ("means %.1f %.1f %.1f " % (wmean,bmean))
#print ("stds  %.1f %.1f %.1f " % (wstds,bstds))



"""""Plotting

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
    
