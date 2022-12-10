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
from matplotlib import pyplot as plt
from PIL import Image

""" Normalize the Image"""
def Filtered_Img_Normalize(image,x,y):
    
  #  norm_img = np.zeros((x,y),np.float32)
    final_img = cv2.normalize(image, None, 0, 15, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    cv2.imwrite('2.1_Normalized_Img.tif', final_img)
   image_size = image.shape

    block1 = int(image_size[1]/20)
    block2 = int(image_size[0]/20)
    Blocks = (block1,block2)

    print('Original Image Size: ',image_size)
    print('Image Size in Block units: ',Blocks)
    cv2.imwrite('2.2_blocks.tif', final_img)
    return final_img

#norm_image = cv2.normalize(raw_img, None, alpha=0, beta=15, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

