# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



from skimage.feature import graycomatrix, graycoprops
from skimage import data
import cv2
import numpy as np 
from matplotlib import pyplot as plt

PATCH_SIZE = 20
distance = [1]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

img = cv2.imread("./breastXray.tif", 
cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

plt.imshow(img,cmap ='gray')


GLCM = graycomatrix(img, distance, angles, levels=256,
                        symmetric=True, normed=True)

 

position_operators = [(100, 500),(300, 400),(400, 300),(450, 100)]
position_patches = []
for loc in position_operators:
    position_patches.append(img[loc[0]:loc[0] + PATCH_SIZE,
                               loc[1]:loc[1] + PATCH_SIZE])

grass_locations = [(280, 454), (342, 223), (444, 192), (455, 455)]
grass_patches = []
for loc in grass_locations:
    grass_patches.append(img[loc[0]:loc[0] + PATCH_SIZE,
                               loc[1]:loc[1] + PATCH_SIZE])
    
for patch in (position_patches + grass_patches):
    glcm = graycomatrix(patch, distances=[5], angles=[0], levels=256,
                        symmetric=True, normed=True)




#GLCM = graycomatrix(img, distance , angles).astype('uint16')

for patch in (position_patches):
    GLCM = graycomatrix(patch, distance, angles, levels=256,
                        symmetric=True, normed=True)
    
plt.figure(figsize=(8,8))
i = 1

for idd,d in enumerate(distance):
    for ida,a in enumerate(angles):
        C = GLCM[:,:,idd,ida]
        plt.subplot(2,2,i)
        plt.imshow(C, vmin=GLCM.min(), vmax=GLCM.max(), cmap=plt.cm.jet) # Give the same scale to all images!
        plt.title('d = %d, a = %.2f'%(d,a))
        i += 1
plt.show()
