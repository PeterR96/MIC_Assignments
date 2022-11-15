import math
import cv2
import numpy as np 
from matplotlib import pyplot as plt
from FCNs_1_3 import *

raw_img = cv2.imread("./Normalized_Img.tif",cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
#gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)                     
x, y = raw_img.shape
bits=2**16
thresh = 23000
Normalized_Img=raw_img
Sobel_Filter = Sobel_kernel(Normalized_Img)

for i in range (40):
    G=Img_Gradient(Sobel_Filter,thresh)
    thresh = thresh + 100
    cv2.imwrite('Treshhold_'+str(i)+'.tif', G)
