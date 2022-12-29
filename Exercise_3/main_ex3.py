# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 14:53:06 2022

@author: Peter Rott, Markus Wernard and Filip Slatkovic
"""

import csv
import numpy as np
from FCNs_3 import *


"""
2.1 Load the data of the “XL_2022.csv” file. While the last column contains the class labels the rest
of the array contains the design matrix X. How many examples does the data include? How
many features?
"""
reader = csv.reader(open("XL_2022.csv", newline = ""), delimiter=",")
x = list(reader)
dataset = np.array(x)
#Featuers = 17 (Columns) , Examples = 676 (Rows)

"""
2.2. Extract the design matrix from the dataframe by dropping the last column (“Label”) and split the
data into a training and a test set. The test set should contain the first and the last 52 examples
(rows) of the data. All other examples should be part of the training set. How big is your test and
training dataset, how many samples from each class do they contain?
"""
class_label = dataset [:, -1]
design_matrix = dataset [:, :-1]

test_data = extract_test_data (design_matrix)
train_data = extract_train_data (design_matrix)