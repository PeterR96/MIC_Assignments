# -*- coding: utf-8 -*-
"""

Created on 8th December 15:52:00 2022

@author: Peter, Filip, Markus
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from PIL import Image
#from FCN_2_1 import *


#variables

#x, y = image.shape
#bits=2**16

#load the image
image = cv2.imread("./breastXray.tif", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
cv2.imwrite('raw_img'+'.tif', image)
#image is the size of 560x480

norm_image = cv2.normalize(image, None, alpha=0, beta=15, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

(image_size) = image.shape

BlocksX = int(image_size[0]/20)
BlocksY = int(image_size[1]/20)
Blocks = (BlocksY,BlocksX)

print('Original Image Size: ',image_size)
print('Image Size in Block units: ',Blocks)
