from FCNs_1_2 import *
import math
import cv2
import numpy as np 
from matplotlib import pyplot as plt

raw_img = cv2.imread("./OCTimage_raw.tif", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
cv2.imshow(raw_img)
