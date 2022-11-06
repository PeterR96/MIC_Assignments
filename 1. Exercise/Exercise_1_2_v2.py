# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 16:45:54 2022

@author: Markus
"""

import cv2
import numpy as np 
from matplotlib import pyplot as plt
oct_image = cv2.imread("./OCTimage_raw.tif", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)


"""2 OCT image preprocessing framework"""

"""2.1 Histogram
# create the histogram
histogram, bin_edges = np.histogram(oct_image, bins=65536, range=(0, 255))

# configure and draw the histogram figure
plt.figure(1)
plt.title(" OCT_Image_raw Histogram")
plt.xlabel("pixel value")
plt.ylabel("count")
plt.xlim([0.0, 255.0])  
plt.plot(bin_edges[0:-1], histogram) """ 


"""2.2 Intensity transformation"""

"""Log Transfomation"""
c = 65535/(np.log(1 + np.max(oct_image)))
log_transformed = c * np.log(1 + oct_image)
  
# Specify the data type.
log_transformed = np.array(log_transformed, dtype = np.uint16)
  
# Save the output.
cv2.imwrite('oct_log_transformed.tif', log_transformed)


"""Gamma Transformation"""
for gamma in [0.2]:
      
    # Apply gamma correction.
    gamma_corrected = np.array(65535*(oct_image / 65535) ** gamma, dtype = 'uint16')
  
    # Save edited images.
    cv2.imwrite('oct_gamma_transformed'+str(gamma)+'.tif', gamma_corrected) 
    
    
"""2.3 Spital Filter"""

n=5
"""Average Filter Gamma"""
kernel = np.ones((n,n),np.float32)/n**2
dst = cv2.filter2D(gamma_corrected,-1,kernel)
cv2.imwrite('oct_gamma_transformed_avg.tif', dst)

"""Gaussian Filter Gamma"""
blur = cv2.GaussianBlur(gamma_corrected, (n,n), 0)
cv2.imwrite('oct_gamma_transformed_gaussian.tif', blur)


"""Average Filter Log"""
kernel1 = np.ones((n,n),np.float32)/n**2
dst1 = cv2.filter2D(log_transformed,-1,kernel)
cv2.imwrite('oct_log_transformed_avg.tif', dst1)

"""Gaussian Filter Log"""
blur1 = cv2.GaussianBlur(log_transformed, (n,n), 0)
cv2.imwrite('oct_log_gaussian.tif', blur1)

"""Plotting"""

titles=['Original','oct_gamma_transformed','Averaging Filter Gamma','Gaussian Filter Gamma',
        'Averaging Filter Log','Gaussian Filter Log']
images=[oct_image,gamma_corrected,dst,blur,dst1,blur1]

plt.figure('images')
for i in range(6):
    plt.subplot(3,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
    
plt.show()

# create the histograms
for j in range(6):
    plt.figure(j)
    histogram, bin_edges = np.histogram(images[j], bins=65536, range=(np.min(images[j]), np.max(images[j])))
    plt.title(titles[j])
    plt.xlabel("pixel value")
    plt.ylabel("count")
    histplot = plt.plot(bin_edges[0:-1], histogram)
    plt.savefig(titles[j]+".jpg", format='jpg')