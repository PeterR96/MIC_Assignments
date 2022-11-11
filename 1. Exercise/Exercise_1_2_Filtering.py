# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 16:45:54 2022

@author: Markus
"""

import cv2
import numpy as np 
from matplotlib import pyplot as plt
oct_image = cv2.imread("./OCTimage_raw.tif", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

x, y = oct_image.shape
bits=2**16

"""2 OCT image preprocessing framework"""
"""2.2 Intensity transformation"""

"""Log Transfomation"""
c = (bits-1)/(np.log(1 + np.max(oct_image)))
log_transformed = c * np.log(1 + oct_image)
  
# Specify the data type.
log_transformed = np.array(log_transformed, dtype = np.uint16)
  
# Save the output.
cv2.imwrite('oct_log_transformed.tif', log_transformed)


"""Gamma Transformation"""
for gamma in [0.2]:
      
    # Apply gamma correction.
    gamma_corrected = np.array((bits-1)*(oct_image / (bits-1)) ** gamma, dtype = 'uint16')
  
    # Save edited images.
    cv2.imwrite('oct_gamma_transformed'+str(gamma)+'.tif', gamma_corrected) 
    
    
"""2.3 Spital Filter"""
n=5

""""Gamma Transformation Filtering"""
#Average Filter Gamma
kernel = np.ones((n,n),np.float32)/n**2
dst = cv2.filter2D(gamma_corrected,-1,kernel)
cv2.imwrite('oct_gamma_transformed_avg.tif', dst)

#Gaussian Filter Gamma
blur = cv2.GaussianBlur(gamma_corrected, (n,n), 0)
cv2.imwrite('oct_gamma_transformed_gaussian.tif', blur)


""" Log Transformation Filtering"""
#Average Filter Log
kernel2 = np.ones((n,n),np.float32)/n**2
dst1 = cv2.filter2D(log_transformed,-1,kernel2)
cv2.imwrite('oct_log_transformed_avg.tif', dst1)

#Gaussian Filter Log
blur1 = cv2.GaussianBlur(log_transformed, (n,n), 0)
cv2.imwrite('oct_log_gaussian.tif', blur1)


""" Normalize the Image"""
norm_img = np.zeros((x,y))
final_img = cv2.normalize(oct_image, norm_img, 0, (bits-1), cv2.NORM_MINMAX)
cv2.imwrite('otc_normalized.jpg', final_img)

#Average Filter Normalized
kernel3 = np.ones((n,n),np.float32)/n**2
dst2 = cv2.filter2D(final_img,-1,kernel3)
cv2.imwrite('oct_norm_avg.tif', dst2)

#Gaussian Filter Normalized
blur2 = cv2.GaussianBlur(final_img, (n,n), 0)
cv2.imwrite('oct_norm_gaussian.tif', blur2)

"""2.4"""

 #21x21 neighborhood
in_image=cv2.imread('oct_norm_gaussian.tif')
pix=in_image[500][500]
#(means, stds) = cv2.meanStdDev(pix)
#wmean,bmean=means.flatten()
#wstds,bstds=stds.flatten()
#print ("means %.1f %.1f %.1f " % (wmean,bmean))
#print ("stds  %.1f %.1f %.1f " % (wstds,bstds))



"""Plotting"""

titles=['Original','oct_gamma_transformed','Averaging Filter Gamma','Gaussian Filter Gamma',
        'Averaging Filter Log','Gaussian Filter Log','oct_norm_avg','oct_norm_gaussian']
images=[oct_image,gamma_corrected,dst,blur,dst1,blur1,dst2,blur2]

plt.figure('images')
for i in range(8):
    plt.subplot(4,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
    
plt.show()

# create the histograms
for j in range(8):
    plt.subplot(4,2,j+1)
    histogram, bin_edges = np.histogram(images[j], bins=bits, range=(0, bits))
    plt.title(titles[j])
    plt.xlabel("pixel value")
    plt.ylabel("count")
    histplot = plt.plot(bin_edges[0:-1], histogram)
    
    
    
    
   #Normalized Histogram 
plt.show()    
histogram, bin_edges = np.histogram(final_img, bins=bits, range=(0, 255))

    # configure and draw the histogram figure
plt.figure(1)
plt.title(" Norm_Image_raw Histogram")
plt.xlabel("pixel value")
plt.ylabel("count")  
plt.plot(bin_edges[0:-1], histogram)

