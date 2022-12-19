# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import cv2

import skimage
import sklearn
from skimage.feature import greycomatrix, greycoprops
from sklearn.cluster import KMeans



PATCH_SIZE = 20
distance = [1]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

img = cv2.imread("./breastXray.tif", 
cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

plt.imshow(img,cmap ='gray')

""" 1.1---------------------------------------------------------------------------------------"""
norm_image = cv2.normalize(img, None, alpha=0, beta=15, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

image_size = img.shape;

BlocksX = int(image_size[1]/20)
BlocksY = int(image_size[0]/20)
Blocks = (BlocksY,BlocksX)

print('Original Image Size: ',image_size)
print('Image Size in Block units: ',Blocks)

""" 1.2---------------------------------------------------------------------------------------"""
region=[];

for i in range(0,image_size[0],20)  :
    row_arr = []
    for j in range(0,image_size[1],20):
        img=norm_image[i:(i+20),j:(j+20)];
        row_arr.append(img)
    region.append(row_arr)
region = np.asarray(region)

glcm_region = []

for i in range(region.shape[0]):
    glcm_row = []
    for j in range(region.shape[1]):
        glcm = greycomatrix(region[i,j],distance, angles, levels=16,symmetric=True);
        #glcm = glcm[:,:,0,0] #getting rid of last two dim
        glcm_row.append(glcm)
    glcm_region.append(glcm_row)
glcm_region = np.asarray(glcm_region)

print('Size of one GLCM: ',glcm_region.shape)

""" 1.3---------------------------------------------------------------------------------------"""
props = ['correlation','contrast','energy','homogeneity']
props_list = []

for k in props:

    props_region = []

    for i in range(glcm_region.shape[0]):
        props_row = []
        for j in range(glcm_region.shape[1]):
            props_vlu = greycoprops(glcm_region[i,j],k)
            props_row.append(props_vlu)
        props_region.append(props_row)
    props_region =np.asarray(props_region)
    
    props_list.append(props_region)
props_list = np.asarray(props_list)

props_list.shape

fig,ax=plt.subplots(4,4,figsize=(12,10))
for i in range(props_list.shape[0]):
    for j in range(props_list.shape[4]):
        ax[i,j].imshow(props_list[j,:,:,0,i],cmap ='gray')
        ax[i,j].title.set_text(props[j]+", Deg: "+str(angles[i]))
fig.tight_layout()

""" 1.4---------------------------------------------------------------------------------------"""
design = []

for i in range(0,4):
    for j in range(0,4):
        liste = props_list[i,:,:,0,j].flatten()
        design.append(liste)
design = np.asarray(design)

design.shape
""" 1.5---------------------------------------------------------------------------------------"""
glcm_region_bonus = []

for i in range(region.shape[0]):
    glcm_row = []
    for j in range(region.shape[1]):
        glcm = greycomatrix(region[i,j],distances=[1,11], angles=[0], levels=16,symmetric=True);
        #glcm = glcm[:,:,0,0] #getting rid of last two dim
        glcm_row.append(glcm)
    glcm_region_bonus.append(glcm_row)
glcm_region_bonus = np.asarray(glcm_region_bonus)


props_region_bonus = []

for i in range(glcm_region.shape[0]):
    props_row = []
    for j in range(glcm_region.shape[1]):
        props_vlu = greycoprops(glcm_region_bonus[i,j],'contrast')
        props_row.append(props_vlu)
    props_region_bonus.append(props_row)
props_region_bonus =np.asarray(props_region_bonus)

fig,ax=plt.subplots(1,2,figsize=(12,10))
for i in range(props_region_bonus.shape[2]):
        ax[i].imshow(props_region_bonus[:,:,i,0,])
       
fig.tight_layout()

"""2.1-----------------------------------------------------------------------------------------"""
X = np.transpose(design)
kmeans = KMeans(n_clusters=4).fit(X)

"""2.2-----------------------------------------------------------------------------------------"""
segm_img = kmeans.labels_.reshape((28,24))
plt.imshow(segm_img)

newMtx=[]
colom = []

for i in range(segm_img.shape[0]):
    row = []
    for j in range(segm_img.shape[1]):
        
        for k in range(20):
            row.append(segm_img[i,j])
        
    for l in range(20):
        colom.append(row)
        
    newMtx.append(colom)

newMtx= np.asarray(colom)

plt.imshow(newMtx)

plt.imsave('Segmentation.png',newMtx)

norm_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
norm_label = cv2.normalize(newMtx, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

added_image = cv2.addWeighted(norm_image,0.9,norm_label,0.4,0)
plt.imshow(added_image)

plt.imsave('Overlay.png',added_image)