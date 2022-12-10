# -*- coding: utf-8 -*-
"""

Created on 8th December 15:52:00 2022

@author: Peter, Filip, Markus
"""

"""
1.1 Load the image “breastXray.tif”. What is the original image size and hence how many 
blocks/regions are processed in the following steps?"
"""

import cv2
import numpy as np
import skimage as skimg
from matplotlib import pyplot as plt
from PIL import Image

""" Normalize the Image"""
def Normalize_Image(img, greylevels):
    norm_img = cv2.normalize(img, None, 0, greylevels-1, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_8U)
    cv2.imwrite('2_1_Normalized_Img.tif', norm_img)
    return norm_img

""" Determine Image Size """
def Image_Size(img):
    image_size = img.shape
    print('Original Image Size: ',image_size)
    return image_size

""" Determine Number of Blocks """
def NumberOfBlocks(img, blocksize_x, blocksize_y):
    image_size = img.shape
    blocks_x = int(image_size[1]/blocksize_x)
    blocks_y = int(image_size[0]/blocksize_y)
    blocks = (blocks_x,blocks_y)
    print('ATTENTION: indirect correlation between rows/columns and Dx/Dy')
    print('Image Size in Block units: ',blocks)
    return blocks

""" Determine GLCM & Calculate Texture Properties """
def GLCM(img, blocks, distances, angles, greylevels):
    glcm = np.zeros([greylevels,greylevels,np.size(distances),np.size(angles),blocks[0],blocks[1]])
    glcm_correlation = np.zeros([1,4,blocks[0],blocks[1]])
    glcm_contrast = np.zeros([1,4,blocks[0],blocks[1]])
    glcm_energy = np.zeros([1,4,blocks[0],blocks[1]])
    glcm_homogeneity = np.zeros([1,4,blocks[0],blocks[1]])
    for Dx in range (0, blocks[0]):
        for Dy in range (0, blocks[1]):
            if np.sum(img[Dx*20:Dx*20+20,Dy*20:Dy*20+20])>0:
                glcm[:,:,:,:,Dx,Dy] = skimg.feature.graycomatrix(img[Dx*20:Dx*20+20,Dy*20:Dy*20+20],
                                                  distances=distances,
                                                  angles=angles,
                                                  levels=greylevels)
                glcm_correlation[:,:,Dx,Dy] = skimg.feature.graycoprops(glcm[:,:,:,:,Dx,Dy], 'correlation')
                glcm_contrast[:,:,Dx,Dy] = skimg.feature.graycoprops(glcm[:,:,:,:,Dx,Dy], 'contrast')
                glcm_energy[:,:,Dx,Dy] = skimg.feature.graycoprops(glcm[:,:,:,:,Dx,Dy], 'energy')
                glcm_homogeneity[:,:,Dx,Dy] = skimg.feature.graycoprops(glcm[:,:,:,:,Dx,Dy], 'homogeneity')
    
    print('Size of each GLCM: (',glcm.shape[0],',',glcm.shape[1],')')            
    return glcm, glcm_correlation, glcm_contrast, glcm_energy, glcm_homogeneity

""" Show processed Images based on descriptors """
def Show_GLCM_Correlation(glcm_correlation, raw_img):
    img_0deg = np.zeros([glcm_correlation.shape[2],glcm_correlation.shape[3]])
    img_45deg = np.zeros([glcm_correlation.shape[2],glcm_correlation.shape[3]])
    img_90deg = np.zeros([glcm_correlation.shape[2],glcm_correlation.shape[3]])
    img_135deg = np.zeros([glcm_correlation.shape[2],glcm_correlation.shape[3]])
    
    for Dx in range (0, glcm_correlation.shape[2]):
        for Dy in range (0, glcm_correlation.shape[3]):
            img_0deg[Dx,Dy] = glcm_correlation[0,0,Dx,Dy]
            img_45deg[Dx,Dy] = glcm_correlation[0,1,Dx,Dy]
            img_90deg[Dx,Dy] = glcm_correlation[0,2,Dx,Dy]
            img_135deg[Dx,Dy] = glcm_correlation[0,3,Dx,Dy]
     
    fig = plt.figure()
    raw = fig.add_subplot(2,3,1)
    raw.imshow(raw_img, cmap=plt.cm.gray, vmin=0, vmax=255)
    raw.set_xlabel('Original Image')
    img1 = fig.add_subplot(2,3,2)
    img1.imshow(img_0deg, cmap=plt.cm.gray, vmin=0, vmax=255)
    img1.set_xlabel('Correlation (direction: 0°, distance: 1)')
    img2 = fig.add_subplot(2,3,3)
    img2.imshow(img_45deg, cmap=plt.cm.gray, vmin=0, vmax=255)
    img2.set_xlabel('Correlation (direction: 45°, distance: 1)')
    img3 = fig.add_subplot(2,3,5)
    img3.imshow(img_90deg, cmap=plt.cm.gray, vmin=0, vmax=255)
    img3.set_xlabel('Correlation (direction: 90°, distance: 1)')
    img4 = fig.add_subplot(2,3,6)
    img4.imshow(img_135deg, cmap=plt.cm.gray, vmin=0, vmax=255)
    img4.set_xlabel('Correlation (direction: 135°, distance: 1)')
    plt.show()
    return img_0deg, img_45deg, img_90deg, img_135deg