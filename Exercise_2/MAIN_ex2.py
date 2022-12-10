########################################################
# Medical Image Computing - Exercise 2
# MAIN_ex2.py
# Authors: Peter Rott, Filip Slatkovic, Markus Wernard
# Date: 10.12.2022
########################################################

# IMPORTS
import cv2
from FCN_2_1 import *

# DEFINITIONS
greylevels = 16
blocksize_x = 20
blocksize_y = 20
distances = [1]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

########################################################
# 1. Image texture descriptors
# Functions can be found in FCN_2_1.py
########################################################

""" 1.1 Load the image “breastXray.tif”. What is the original image size and hence 
        how many blocks/regions are processed in the following steps?"""
raw_img = cv2.imread("./breastXray.tif", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

# Normalize Image
norm_img = Normalize_Image(raw_img, greylevels)
# Determine Image Size
image_size = Image_Size(norm_img)
# Determine Number of Blocks
blocks = NumberOfBlocks(norm_img, blocksize_x, blocksize_y)

""" 1.2. Determine the gray level co-occurrence matrix (GLCM) for each region/block 
         using a position operator [Dx, Dy]. Choose a distance, e.g. D = 1, and 
         extract the GLCM at four different directions [0°, 45°, 90°, 135°]. 
         The number of gray levels should be set to 16. What is the size of one GLCM?"""
 
glcm = GLCM(norm_img, blocks, distances, angles, greylevels)[0]

""" 1.3. Calculate the correlation, contrast, energy and homogeneity for each GLCM. Show processed
         images based on the descriptors, i.e. Correlation at four directions with D = 1, 
         Contrast at fourdirections with D =1, etc., using subplots (see Figure 1). 
         How many features did you extract?
         Comment on the differences between the different texture descriptors."""
        
glcm_correlation = GLCM(norm_img, blocks, distances, angles, greylevels)[1]
glcm_contrast = GLCM(norm_img, blocks, distances, angles, greylevels)[2]
glcm_energy = GLCM(norm_img, blocks, distances, angles, greylevels)[3]
glcm_homogeneity =GLCM(norm_img, blocks, distances, angles, greylevels)[4]

img_0deg = Show_GLCM_Correlation(glcm_correlation, raw_img)[0]
img_45deg = Show_GLCM_Correlation(glcm_correlation, raw_img)[1]
img_90deg = Show_GLCM_Correlation(glcm_correlation, raw_img)[2]
img_135deg = Show_GLCM_Correlation(glcm_correlation, raw_img)[3]