# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 19:36:41 2022

@author: Peter, Filip, Markus
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import cv2

image = cv2.imread("./breastXray.tif", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

norm_image = cv2.normalize(image, None, alpha=0, beta=15, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

image_size = image.shape;

BlocksX = int(image_size[1]/20)
BlocksY = int(image_size[0]/20)
Blocks = (BlocksY,BlocksX)

print('Original Image Size: ',image_size)
print('Image Size in Block units: ',Blocks)