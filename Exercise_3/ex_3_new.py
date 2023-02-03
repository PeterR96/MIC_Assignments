# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 18:51:15 2023

@author: Peter Rott
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from FCNs_3 import *

#import cv2
import csv
import pandas as pd
from numpy import genfromtxt
import seaborn as sn

import skimage
import sklearn
from skimage.feature import greycomatrix, greycoprops
from sklearn.cluster import KMeans

"""
2.1 Load the data of the “XL_2022.csv” file. While the last column contains the class labels the rest
of the array contains the design matrix X. How many examples does the data include? How
many features?
"""

data_labeld = pd.read_csv('XL_2022.csv')
data = genfromtxt('XL_2022.csv', delimiter=',')

"""
2.2. Extract the design matrix from the dataframe by dropping the last column (“Label”) and split the
data into a training and a test set. The test set should contain the first and the last 52 examples
(rows) of the data. All other examples should be part of the training set. How big is your test and
training dataset, how many samples from each class do they contain?
"""

design_matrix = data[1:677,0:16]
test_data = extract_test_data (design_matrix)
size_test_data= np.shape(test_data)
train_data = extract_train_data (design_matrix)
size_train_data= np.shape(train_data)

"""
2.3. Calculate and plot the covariance matrix of the features and visualize the result from your
training dataset. Based on your visualization, select a reduced set of features you think might be
useful for further classification and generate a reduced design matrix X1. Give a rationale for
your choice of features.
"""
# calculate the covariance matrix
cov_matrix = np.cov(np.transpose(train_data))

# visualize the result
plt.imshow(cov_matrix, cmap='hot', interpolation='nearest')
plt.show()

index = [0,4,6,8,12,14] 
index.append(16) #add Labels
selection = data_labeld.columns[index]
cov_selected = data_labeld [selection].head()
data_X1_df=data_labeld[selection]
data_X1 = np.asarray(data_X1_df)
design_matrix_X1 = data_X1[0:676,0:6]


"""
3.1. Use a kNN classifier to classify the data into 4 groups using 5 nearest neighbors. Analyze the
performance of the classifier using the test dataset by calculating the classification error as the
number of false classifications divided by the total number of samples. Compare the
performance of the classifier when applied to the data containing all features X and to your
reduced design matrix X1.
"""