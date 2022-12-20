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

#from skimage.feature import greycomatrix, greycoprops
from skimage.feature import graycomatrix, graycoprops



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
            glcm_correlation[:,:,Dy,Dx] = graycoprops(glcm[:,:,:,:,Dy,Dx], 'correlation')
            glcm_contrast[:,:,Dy,Dx] = graycoprops(glcm[:,:,:,:,Dy,Dx], 'contrast')
            glcm_energy[:,:,Dy,Dx] = graycoprops(glcm[:,:,:,:,Dy,Dx], 'energy')
            glcm_homogeneity[:,:,Dy,Dx] = graycoprops(glcm[:,:,:,:,Dy,Dx], 'homogeneity')
                            
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
    fig.savefig('2_1_3_'+str(title)+'distance_1.tif',dpi=1000,bbox_inches='tight')
  
""""Build Design Matrix"""   
def design_matrix(glcm_correlation, glcm_contrast,glcm_energy,glcm_homogeneity):
    
    design_matrix = np.zeros((672,16))
    D_x = -1
    D_y = 0
    i=-1               
    for x in range (672):
        if D_x == 23:
            D_y = D_y+1
            D_x = 0
            
        else:
            D_x = D_x + 1
            
            for y in range (16):
                
                if y<=3:
                    i=i+1
                    design_matrix[x,y] =  glcm_correlation[0, i, D_y, D_x]                                                         
                    if (i==3):
                        i=-1 
                                       
                if y>3 and y<=7:
                    i=i+1
                    design_matrix[x,y] = glcm_contrast[0, i, D_y, D_x]                    
                    if (i==3):
                        i=-1      
                        
                if y>7 and y<=11:
                    i=i+1
                    design_matrix[x,y] = glcm_energy[0, i, D_y, D_x]
                    if (i==3):
                        i=-1     
                        
                if y>11 and y<=16:
                    i=i+1
                    design_matrix[x,y] = glcm_homogeneity[0, i, D_y, D_x]
                    if (i==3):
                        i=-1 
    
    #design_matrix.shape
    img_M=np.transpose(design_matrix)
    plt.figure()
    plt.title("Design_matrix")  
    plt.ylabel("Features")
    plt.xlabel("Observations")     
    plt.imshow(img_M,cmap=plt.cm.gray)
    plt.savefig('2_1_4_Design_Matrix.tif',dpi=300, bbox_inches='tight')
    return design_matrix

"""Kmeans clustering"""
def kmeansclustering(design_matrix):    
    kmeans = KMeans(n_clusters=4, n_init=10).fit(design_matrix)
    return kmeans

"""Kmeans Visualization"""
def Vkmeans(kmeans, raw_img):
        label_img = kmeans.labels_.reshape(28,24)
        label_img = cv2.resize(label_img, (460,560),interpolation=cv2.INTER_NEAREST)
        
        for i in range (20):
            label_img[:,i]= label_img[0,21]
            
        plt.figure()
        plt.title("Overlay with distance: 1 / Cluster = 4")  
        plt.imshow(raw_img, cmap="gray")
        plt.imshow(label_img, cmap="jet", alpha=0.5)
        plt.savefig('2_2_2_Overlay_distance_1.tif',dpi=1000, bbox_inches='tight')
        plt.show()
        return label_img
    

def Show_GLCM_Descriptor_d3(glcm_descriptor, glcm_descriptor_d3, title, raw_img):
   # create figure
    fig = plt.figure()
    
   # display direction 0°
    img0deg = fig.add_subplot(2,4,1)
    img0deg.imshow(glcm_descriptor[0,0,:,:],cmap=plt.cm.gray, interpolation='none', vmin=0, vmax=np.max(glcm_descriptor[0,0,:,:]))
    img0deg.set_title('dir: 0°, dis: 1')
    img0deg.set_xticks([])
    img0deg.set_yticks([])
   # display direction 45°
    img45deg = fig.add_subplot(2,4,2)
    img45deg.imshow(glcm_descriptor[0,1,:,:],cmap=plt.cm.gray,interpolation='none', vmin=0, vmax=np.max(glcm_descriptor[0,1,:,:]))
    img45deg.set_title('dir: 45°, dis: 1')
    img45deg.set_xticks([])
    img45deg.set_yticks([])
   # display direction 90°
    img90deg = fig.add_subplot(2,4,3)
    img90deg.imshow(glcm_descriptor[0,2,:,:],cmap=plt.cm.gray,  interpolation='none', vmin=0, vmax=np.max(glcm_descriptor[0,2,:,:]))
    img90deg.set_title('dir: 90°, dis: 1')
    img90deg.set_xticks([])
    img90deg.set_yticks([])
   # display direction 135°
    img135deg = fig.add_subplot(2,4,4)
    img135deg.imshow(glcm_descriptor[0,3,:,:], cmap=plt.cm.gray, interpolation='none', vmin=0, vmax=np.max(glcm_descriptor[0,3,:,:]))
    img135deg.set_title('dir: 135°, dis: 1')
    img135deg.set_xticks([])
    img135deg.set_yticks([])
    
    # display direction 0°
    img0deg_d3 = fig.add_subplot(2,4,5)
    img0deg_d3.imshow(glcm_descriptor_d3[0,0,:,:],cmap=plt.cm.gray, interpolation='none', vmin=0, vmax=np.max(glcm_descriptor_d3[0,0,:,:]))
    img0deg_d3.set_title('dir: 0°, dis: 3')
    img0deg_d3.set_xticks([])
    img0deg_d3.set_yticks([])
   # display direction 45°
    img45deg_d3 = fig.add_subplot(2,4,6)
    img45deg_d3.imshow(glcm_descriptor_d3[0,1,:,:],cmap=plt.cm.gray,interpolation='none', vmin=0, vmax=np.max(glcm_descriptor_d3[0,1,:,:]))
    img45deg_d3.set_title('dir: 45°, dis: 3')
    img45deg_d3.set_xticks([])
    img45deg_d3.set_yticks([])
   # display direction 90°
    img90deg_d3 = fig.add_subplot(2,4,7)
    img90deg_d3.imshow(glcm_descriptor_d3[0,2,:,:],cmap=plt.cm.gray,  interpolation='none', vmin=0, vmax=np.max(glcm_descriptor_d3[0,2,:,:]))
    img90deg_d3.set_title ('dir: 90°, dis: 3')
    img90deg_d3.set_xticks([])
    img90deg_d3.set_yticks([])
   # display direction 135°
    img135deg_d3 = fig.add_subplot(2,4,8)
    img135deg_d3.imshow(glcm_descriptor_d3[0,3,:,:], cmap=plt.cm.gray, interpolation='none', vmin=0, vmax=np.max(glcm_descriptor_d3[0,3,:,:]))
    img135deg_d3.set_title('dir: 135°, dis: 3')
    img135deg_d3.set_xticks([])
    img135deg_d3.set_yticks([])   
   # display output
    plt.suptitle(title)
    plt.tight_layout()  
    plt.show()
    fig.savefig('2_1_3_'+str(title)+'distance_1.tif',dpi=300, bbox_inches='')