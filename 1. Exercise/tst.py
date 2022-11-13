import math
import cv2
import numpy as np 
from matplotlib import pyplot as plt

raw_img = cv2.imread("./Normalized_Img.tif",cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
#gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)                     
x, y = raw_img.shape


gX = cv2.Sobel(raw_img, ddepth=cv2.CV_32F, dx=1, dy=0)
gY = cv2.Sobel(raw_img, ddepth=cv2.CV_32F, dx=0, dy=1)
mag = np.sqrt((gX ** 2) + (gY ** 2))
orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180

#cv2.imshow("gradient magnitude", mag)
cv2.imshow("gx", gX)
#cv2.imshow("by", gY)

cv2.waitKey(0)
