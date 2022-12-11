###############################################################################
# Medical Image Computing - Exercise 2
# MAIN_ex2.py
# Description: Functions from the Tasks are called to give a better overview
# Authors: Peter Rott, Filip Slatkovic, Markus Wernard
# Date: 11.12.2022
###############################################################################

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
 
glcm = Calc_GLCM(norm_img, blocks, distances, angles, greylevels)

""" 1.3. Calculate the correlation, contrast, energy and homogeneity for each GLCM. Show processed
         images based on the descriptors, i.e. Correlation at four directions with D = 1, 
         Contrast at four directions with D=1, etc., using subplots (see Figure 1). 
         How many features did you extract?
         Comment on the differences between the different texture descriptors."""
        
# Calculate GLCM Descriptors
glcm_correlation = Calc_GLCM_Descriptors(glcm, blocks)[0]
glcm_contrast = Calc_GLCM_Descriptors(glcm, blocks)[1]
glcm_energy = Calc_GLCM_Descriptors(glcm, blocks)[2]
glcm_homogeneity = Calc_GLCM_Descriptors(glcm, blocks)[3]

# Show processed images
Show_GLCM_Descriptor(glcm_correlation, 'GLCM Correlation', norm_img)
Show_GLCM_Descriptor(glcm_contrast, 'GLCM Contrast', norm_img)
Show_GLCM_Descriptor(glcm_energy, 'GLCM Energy', norm_img)
Show_GLCM_Descriptor(glcm_homogeneity, 'GLCM Homogeneity', norm_img)

""" 1.4 Build a design matix based on your blocks where each block/region in an observation(sample) 
        and the texture description are the features. What is the size of your design matrix?"""
        