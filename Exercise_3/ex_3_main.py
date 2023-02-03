"""
Created on Wed Feb  1 18:51:15 2023

@author: Peter Rott, Filip Slatkovic and Markus Wernard
"""
import numpy as np
import matplotlib.pyplot as plt
from FCNs_3 import *
import cv2 as cv
import pandas as pd
from numpy import genfromtxt
import seaborn as sn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

"""
2.1 Load the data of the “XL_2022.csv” file. While the last column contains the class labels the rest
of the array contains the design matrix X. How many examples does the data include? How
many features?
"""
data_X = pd.read_csv('XL_2022.csv')
data = genfromtxt('XL_2022.csv', delimiter=',')

"""
2.2. Extract the design matrix from the dataframe by dropping the last column (“Label”) and split the
data into a training and a test set. The test set should contain the first and the last 52 examples
(rows) of the data. All other examples should be part of the training set. How big is your test and
training dataset, how many samples from each class do they contain?
"""
#design matrix and train/test data as array (float 64)
design_matrix = data[1:677,0:16]
train_data_X_arr = extract_train_data (design_matrix)
test_data_X_arr = extract_test_data (design_matrix)

#train_data_X in DataFrame
train_data_X = data_X.iloc[52:(676-52)];

#test_data_X in DataFrame
test_data_X_first_52 = data_X.iloc[0:52];
test_data_X_last_52 =  data_X.iloc[676-52:676];
test_data_X = pd.concat([test_data_X_first_52, test_data_X_last_52])

"""
2.3. Calculate and plot the covariance matrix of the features and visualize the result from your
training dataset. Based on your visualization, select a reduced set of features you think might be
useful for further classification and generate a reduced design matrix X1. Give a rationale for
your choice of features.
"""
# calculate the covariance matrix
cov_matrix = np.cov(np.transpose(train_data_X_arr))

# visualize the result
plt.imshow(cov_matrix, cmap='hot', interpolation='nearest')
plt.show()

#reduced set of features
index = [0,4,6,8,12,14] 
index.append(16) #add lables
selection = data_X.columns[index]
cov_selected = data_X [selection].head()
data_X1= data_X[selection]
data_X1_arr = np.asarray(data_X1)

#reduced design matrix
design_matrix_X1 = data_X1_arr[0:676,0:6]

"""
3.1. Use a kNN classifier to classify the data into 4 groups using 5 nearest neighbors. Analyze the
performance of the classifier using the test dataset by calculating the classification error as the
number of false classifications divided by the total number of samples. Compare the
performance of the classifier when applied to the data containing all features X and to your
reduced design matrix X1.
"""
#train_data_X1
train_data_X1 = data_X1.iloc[52:(676-52)];

#test_data_X1
test_data_X1_first_52 = data_X1.iloc[0:52];
test_data_X1_last_52 =  data_X1.iloc[676-52:676];
test_data_X1 = pd.concat([test_data_X1_first_52, test_data_X1_last_52])

#classifier for X
classifier_X = KNeighborsClassifier(n_neighbors=5)
features = train_data_X.loc[:, train_data_X.columns != 'Label'];
label = train_data_X.loc[:, 'Label'];
classifier_X.fit(features,label)

X_test = test_data_X.loc[:, test_data_X.columns != 'Label'];
y_test = test_data_X.loc[:, 'Label']
y_pred = classifier_X.predict(X_test)

acc_X = metrics.accuracy_score(y_test, y_pred)

#classifier for X1
classifier_X1 = KNeighborsClassifier(n_neighbors=5)
features_X1 = train_data_X1.loc[:, train_data_X1.columns != 'Label'];
label_X1 = train_data_X1.loc[:, 'Label'];
classifier_X1.fit(features_X1,label_X1)

X1_test = test_data_X1.loc[:, test_data_X1.columns != 'Label'];
y1_test = test_data_X1.loc[:, 'Label']
y1_pred = classifier_X1.predict(X1_test)

acc_X1 = metrics.accuracy_score(y1_test, y1_pred)

"""
3.2 Train a random forest on the training data based on the design matrix X. Analyze and interpret
the influence of the number of trees using the out-of-bag classification error (oob_score).
Control the random number generation to ease interpretation (Matlab function rng; sklearn
random forest random_state=0). Evaluate the performance of the random forest on the test
dataset using two different numbers of decision trees in your ensemble. Again, use the predict
function to get the label predictions.
"""
for i in [1,2,3,4,7,13,18,25]:
    rfc=RandomForestClassifier(n_estimators=i)
    rfc.fit(features,label)

    yRF_prediction = rfc.predict(X_test)
    print(f"Accuracy of {i} trees:",metrics.accuracy_score(y_test, yRF_prediction))

sn.set(font_scale=1)
clf = RandomForestClassifier(warm_start=True, oob_score=True,random_state=0)
error = []
min_estimators = 3
max_estimators = 300

for i in range(min_estimators, max_estimators +1, 5):
    clf.set_params(n_estimators=i)
    clf.fit(features,label)
    
    # Record the OOB error for each `n_estimators=i` setting.
    oob_error = 1 - clf.oob_score_
    error.append((i, oob_error))
    
# Generate "OOB error rate" vs. "n_estimators" plot
x,y = zip(*error)
plt.plot(x,y)
plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.show()

"""
3.3 Analyze and interpret the importance of the various features using
rf.OOBPermutedPredictorDeltaError (Matlab) / RandomForestClassifier.feature_importances_
(Python sklearn). Which features were most valuable for the random forest?
"""
feature_names = features.columns;
rfc=RandomForestClassifier(n_estimators=i,warm_start=True, oob_score=True,random_state=0)
rfc.fit(features,label)
feature_importance = pd.Series(rfc.feature_importances_,index=features.columns).sort_values(ascending=False)
feature_importance;

# Creating a bar plot
sn.barplot(x=feature_importance, y=feature_importance.index)
sn.set(font_scale=1)
# Add labels to the graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")

plt.show()

"""
4.1. The design matrix used in the previous sections is based on the image “patches.tif”. Use the
image attached and write a code to extract two more features of your choice for the 20 x 20
pixel sized patches. Re-use the functions you have used in exercise 2.
"""
image = cv.imread("./patches.tif", cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)

plt.imshow(image)
norm_image = cv.normalize(image, None, alpha=0, beta=15, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
"""
4.2. Include the features in the design matrix and re-evaluate the performance of the random forest
using your new design matrix. Discuss the influence of your added features.
"""