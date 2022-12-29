# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 15:40:21 2022

@author: Peter Rott
"""
import numpy as np


def extract_test_data (design_matrix):
    
    first_row = design_matrix[:2,:]
    last_52_rows = design_matrix [-52:, :]
    
    test_data = np.vstack ((first_row, last_52_rows))
    
    return test_data

def extract_train_data (design_matrix):
    
    first_row = design_matrix[:1,:]
    rows_2_to_624 = design_matrix[2:625, :]
    
    train_data = np.vstack ((first_row, rows_2_to_624))
    
    return train_data