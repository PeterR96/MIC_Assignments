###############################################################################
# Medical Image Computing - Exercise 2
# FCN_2_1.py
# Description: Includes functions for Task 1 of Exercise 2
# Authors: Peter Rott, Filip Slatkovic, Markus Wernard
# Date: 11.12.2022
###############################################################################

import cv2
import numpy as np
import skimage as skimg
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
import sklearn

from skimage.feature import greycomatrix, greycoprops



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

""" Calculate GLCM """
def Calc_GLCM(img, blocks, distances, angles, greylevels):
    glcm = np.zeros([greylevels,greylevels,np.size(distances),np.size(angles),blocks[1],blocks[0]])
    for Dy in range (0, blocks[1]):
        for Dx in range (0, blocks[0]):
            if np.sum(img[Dy*20:Dy*20+19,Dx*20:Dx*20+19])>0:
                glcm[:,:,:,:,Dy,Dx] = skimg.feature.graycomatrix(img[Dy*20:Dy*20+19,Dx*20:Dx*20+19],
                                                  distances=distances,
                                                  angles=angles,
                                                  levels=greylevels)
    print('Size of each GLCM: (',glcm.shape[1],',',glcm.shape[0],')')            
    return glcm

""" Calculate Texture Descriptors """
def Calc_GLCM_Descriptors(glcm, blocks):
    glcm_correlation = np.zeros([1,4,blocks[1],blocks[0]])
    glcm_contrast = np.zeros([1,4,blocks[1],blocks[0]])
    glcm_energy = np.zeros([1,4,blocks[1],blocks[0]])
    glcm_homogeneity = np.zeros([1,4,blocks[1],blocks[0]])
    
    for Dy in range (0, blocks[1]):      
        for Dx in range (0, blocks[0]):
            glcm_correlation[:,:,Dy,Dx] = skimg.feature.graycoprops(glcm[:,:,:,:,Dy,Dx], 'correlation')
            glcm_contrast[:,:,Dy,Dx] = skimg.feature.graycoprops(glcm[:,:,:,:,Dy,Dx], 'contrast')
            glcm_energy[:,:,Dy,Dx] = skimg.feature.graycoprops(glcm[:,:,:,:,Dy,Dx], 'energy')
            glcm_homogeneity[:,:,Dy,Dx] = skimg.feature.graycoprops(glcm[:,:,:,:,Dy,Dx], 'homogeneity')
                            
    return glcm_correlation, glcm_contrast, glcm_energy, glcm_homogeneity

""" Show processed Images based on correlation descriptor """
def Show_GLCM_Descriptor(glcm_descriptor, title, raw_img):
   # create figure
    fig = plt.figure()
   # display original image
    raw = fig.add_subplot(2,3,1)
    raw.imshow(raw_img, cmap=plt.cm.gray, vmin=0, vmax=15)
    raw.set_xlabel('Original Image')
    raw.set_xticks([])
    raw.set_yticks([])
   # display direction 0°
    img0deg = fig.add_subplot(2,3,2)
    img0deg.imshow(glcm_descriptor[0,0,:,:],cmap=plt.cm.gray, interpolation='none', vmin=0, vmax=np.max(glcm_descriptor[0,0,:,:]))
    img0deg.set_xlabel('direction: 0°, distance: 1')
    img0deg.set_xticks([])
    img0deg.set_yticks([])
   # display direction 45°
    img45deg = fig.add_subplot(2,3,3)
    img45deg.imshow(glcm_descriptor[0,1,:,:],cmap=plt.cm.gray,interpolation='none', vmin=0, vmax=np.max(glcm_descriptor[0,1,:,:]))
    img45deg.set_xlabel('direction: 45°, distance: 1')
    img45deg.set_xticks([])
    img45deg.set_yticks([])
   # display direction 90°
    img90deg = fig.add_subplot(2,3,5)
    img90deg.imshow(glcm_descriptor[0,2,:,:],cmap=plt.cm.gray,  interpolation='none', vmin=0, vmax=np.max(glcm_descriptor[0,2,:,:]))
    img90deg.set_xlabel('direction: 90°, distance: 1')
    img90deg.set_xticks([])
    img90deg.set_yticks([])
   # display direction 135°
    img135deg = fig.add_subplot(2,3,6)
    img135deg.imshow(glcm_descriptor[0,3,:,:], cmap=plt.cm.gray, interpolation='none', vmin=0, vmax=np.max(glcm_descriptor[0,3,:,:]))
    img135deg.set_xlabel('direction: 135°, distance: 1')
    img135deg.set_xticks([])
    img135deg.set_yticks([])
   # display output
    plt.suptitle(title)
    plt.tight_layout()
    
    plt.show()
    fig.savefig('2_1_3_'+str(title)+'.tif',dpi=1000,bbox_inches='tight')
    
def design_matrix(glcm_correlation, glcm_contrast,glcm_energy,glcm_homogeneity):
    
    design_matrix = np.zeros((672,16))
    D_x = -1
    D_y = 0
    i=0                
    for x in range (672):
        if D_x == 23:
            D_y = D_y+1
            D_x = 0
            
        else:
            D_x = D_x + 1
            
            for y in range (16):
                
                if y<=3:
                    design_matrix[x][y] =  glcm_correlation[0, i, D_y, D_x]                 
                    i=i+1
                    
                    if (i==3):
                        i=0                                        
                if y>3 and y<=7:
                    design_matrix[x][y] = glcm_contrast[0, i, D_y, D_x]
                    i=i+1
                    if (i==3):
                        i=0                        
                if y>7 and y<=11:
                    design_matrix [x][y] = glcm_energy[0, i, D_y, D_x]
                    i=i+1
                    if (i==3):
                        i=0                    
                if y>11 and y<16:
                    design_matrix [x][y] = glcm_homogeneity[0, i, D_y, D_x]
                    i=i+1
                    if (i==3):
                        i=0
    
    design_matrix = design_matrix.transpose()
    design_matrix.shape
    plt.figure()
    plt.title("Design_matrix")  
    plt.ylabel("Features")
    plt.xlabel("Observations")  
    
   
    plt.imshow(design_matrix,cmap=plt.cm.gray)
    plt.savefig('2_1_4_Design_Matrix.tif',dpi=1000, bbox_inches='tight')
   
    return design_matrix

def kmeansclustering(design_matrix):
    X = np.transpose(design_matrix)
    kmeans = KMeans(n_clusters=4, n_init=10).fit(X)
    
    return kmeans

def kmeansVisualize(kmeans,raw_img):
   
    segm_img = kmeans.labels_.reshape((28,24))
    
    plt.figure()
    plt.title("Segmentation")  
    plt.imshow(segm_img)
    plt.savefig('Segmentation.tif',dpi=1000, bbox_inches='tight')
    
    newMtx=[]
    colom = []

    for i in range(segm_img.shape[0]):
        row = []
        for j in range(segm_img.shape[1]):
            
            for k in range(19):
                row.append(segm_img[i,j])
            
        for l in range(19):
            colom.append(row)
            
        newMtx.append(colom)

    newMtx= np.asarray(colom)  
    
    plt.figure()
    plt.title("Overlay")                     
    plt.imshow(raw_img, cmap="gray")
    plt.imshow(newMtx,cmap ='jet', alpha=0.5)
    plt.savefig('Overlay.tif',dpi=1000, bbox_inches='tight')
    
    return newMtx

def design_matrix_hor(glcm_correlation, glcm_contrast,glcm_energy,glcm_homogeneity):
    
    design_matrix_hor = np.zeros((16,672))
    D_x = 0
    D_y = -1
    i=0
             
    for x in range (672):
       if D_y == 27:
           D_x = D_x + 1
           D_y = 0
           
       else:
           D_y = D_y + 1
           
           for y in range (16):
               
               if y<=3:
                   design_matrix_hor[y][x] =  glcm_correlation[0, i, D_y, D_x]                 
                   i=i+1
                   
                   if (i==3):
                       i=0 
                                       
               if y>3 and y<=7:
                   design_matrix_hor[y][x] = glcm_contrast[0, i, D_y, D_x]
                   i=i+1
                   if (i==3):
                       i=0    
                       
               if y>7 and y<=11:
                   design_matrix_hor [y][x] = glcm_energy[0, i, D_y, D_x]
                   i=i+1
                   if (i==3):
                       i=0   
                       
               if y>11 and y<=16:
                   design_matrix_hor [y][x] = glcm_homogeneity[0, i, D_y, D_x]
                   i=i+1
                   if (i==3):
                       i=0
                
    
    
    plt.figure()
    
    plt.title("Design_matrix_hor")      
    plt.ylabel("Features")
    plt.xlabel("Observations")      
    plt.imshow(design_matrix_hor,cmap=plt.cm.gray)
    plt.savefig('2_1_4_Design_Matrix_hor.tif',dpi=1000)
    
    return design_matrix_hor
    
def Vkmeans(kmeans, raw_img):
        label_img = np.asarray(kmeans.labels_).reshape(28,24)
        label_img = cv2.resize(label_img, (480, 560), interpolation=cv2.INTER_NEAREST)
        plt.figure()
        plt.title("Overlay")  
        plt.imshow(raw_img, cmap="gray")
        plt.imshow(label_img, cmap="jet", alpha=0.5)
        plt.show()
        return label_img