#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 04:36:54 2019

@author: ayf

Example:
-------
The 1st Classification Algorithm: K-Nearest Neighbor

Train-Test split
"""

# import modules
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


# invoke this method on ipython console when there are lots of plots
def closeAll():
    plt.close("all")

# method for reading datasets
def read_dataset(path, label_column):
    '''
    label_column; in which column data labels (classes) are placed
    '''
    # make dataset global
    global dataset
    # load csv data set from file using pandas
    dataset = pd.read_csv(path) # the type of dataset is pandas frame 
    # check Variable explorer and see data table
    
    # what is the name of columns (features)
    features = dataset.columns.tolist()
    # remove the label column
    features.remove(label_column)
    
    # we can extract data labels as follows
    labels = dataset.loc[:, label_column] # select label column
    # we can extract the actual data as follows
    data = dataset.loc[:, features] # select all columns except the label column
    # return the data and labels seperately
    return data, labels

# method for displaying data set with labels
def plot_dataset(data, col1, col2, labels, marker, title=""):
    '''
    type of data and labels parameters are "pandas frame" and "pandas series"
    col1 and col2 are the features which will be scattered 
    '''
    # first, get the corresponding columns from the dataset
    data = data.loc[:, [col1, col2]] # get all rows for only col1 and col2
    
    # create a figure
    plt.figure()
    plt.title(title, size=20)
    # scatter each unique group with a different color
    for label in sorted(labels.unique()):
        # get data points belonging to the group with 'label' labeled
        label_points = data[labels==label].values
        # get x and y points
        x_points = label_points[:,0]
        y_points = label_points[:,1]
        # determine label text in figure legend
        label_text = "class: {}".format(label)
        # scatter datapoints belonging to the 'label'
        plt.scatter(x_points, y_points, marker=marker, alpha=0.70, label=label_text)
    
    # place legend box at the best location
    plt.legend(loc="best", fontsize=14)
    # display labels for x and y axes
    plt.xlabel(col1, size=14)
    plt.ylabel(col2, rotation=0, size=14)

# method for displaying class boundaries
def plot_boundary(model):
    # determine bounds of meshgrid 
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    
    # set step size
    h = .05  # step size in the mesh grid (if h is too low, then there might be memory error !)
    # generate a mesh grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # convert mesh grid to vector
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    # predict the label of each data points in grid points
    Z = model.predict(grid_points)
    # reshape Z so it will be a grid again
    Z = Z.reshape(xx.shape)
    # plot grid points with their predicted labels (by KNN) that corresponds to a color
    plt.imshow(Z, interpolation="gaussian", zorder=-100, 
               alpha=0.3, extent=[x_min, x_max, y_min, y_max], 
               aspect="auto", origin="lower")


#%% load intermixed dataset
title =  "binary_intermixed"
path = "../datasets/toy/{}.csv".format(title)

print("::::: Binary Labeled Intermixed Dataset :::::\n")
data, labels = read_dataset(path, "label")
# print data shape 
print(">>> Data shape: {}".format(data.shape))
# how many labels for each class
for label in np.unique(labels):
    count = labels[labels==label].shape[0]
    print("  > {} instances for class {}".format(count, label))
print("")
# plot complete data set
plot_dataset(data, "f0", "f1", labels, "o", title)
# get axes limits
xlim, ylim = plt.xlim(), plt.ylim()

#%% split data set into train and test sets
validation_size = 0.30 # the ratio for train-test split
random_state    = 1111 # control randomness (set to None for full randomness)
# split dataset
trainData, testData, trainLabels, testLabels = train_test_split(data, labels, 
                                                                test_size=validation_size, 
                                                                random_state=random_state,
                                                                shuffle=True)
# take a look at how train set looks like
title1 = title+" (train)"
# print data shape 
print(">>> Train data shape: {}".format(trainData.shape))
# how many labels for each class
for label in np.unique(labels):
    count = trainLabels[labels==label].shape[0]
    print("  > {} instances for class {}".format(count, label))
print("")
# plot train data 
plot_dataset(trainData, "f0", "f1", trainLabels, "o", title1)
# set axes limits
_ = plt.xlim(xlim), plt.ylim(ylim)

# take a look at how test set looks like
title2 = title+" (test)"
# print data shape  
print(">>> Test data shape: {}".format(testData.shape))
# how many labels for each class
for label in np.unique(labels):
    count = testLabels[labels==label].shape[0]
    print("  > {} instances for class {}".format(count, label))
print("")
plot_dataset(testData, "f0", "f1", testLabels, "x", title2)
# set axes limits
_ = plt.xlim(xlim), plt.ylim(ylim)

#%% create a classification model: KNN

# model hyper-parameters (values are up to you)
n_neigbors = 5 # how many neighbors will be checked during the prediction
minkowski_param = 2 # if p=1 then distance metric will be Manhattan, if p=2 than distance metric will be Euclidean

# create the model using the hyper parameters defined above
knn = KNeighborsClassifier(n_neighbors=n_neigbors, 
                           metric='minkowski', p=minkowski_param, 
                           n_jobs=3)

#%% training KNN and evaluting its performance on train set

# train (fit) the model with TRAIN set
print(">>> Model is learning the training set...")
knn.fit(trainData, trainLabels)

# plot training data
plot_dataset(trainData, "f0", "f1", trainLabels, "o", title1)
# set axes limits
_ = plt.xlim(xlim), plt.ylim(ylim)
# plot decision boundary for train data
plot_boundary(knn)


# make predictions on TRAIN set
print(">>> Model is predicting labels of training set...")
predicted_trainset_labels = knn.predict(trainData)

# Evaluating KNN performance on TRAIN set
acc_score = accuracy_score(trainLabels, predicted_trainset_labels)
print(">>> Accuracy score for train set: {}".format(acc_score))
report = classification_report(trainLabels, predicted_trainset_labels)
print(">>> Classification report for train set: \n{}".format(report))
print("")

# what did we do so far ?
# 1. we splitted dataset > train data and test data
# 2. we created a model: KNN
# 3. we let the KNN learn the train data set (data + labels)
# 4. we evaluate KNN performance on train set
# 5. what about test set ? Lets check out KNN performance on test set which KNN did not learn 


#%% evaluting KNN performance on test set

# plot test data
plot_dataset(testData, "f0", "f1", testLabels, "x", title2)
# set axes limits
_ = plt.xlim(xlim), plt.ylim(ylim)
# plot decision boundary for train data
plot_boundary(knn)


# now, make predictions on TEST set
print(">>> Model is predicting labels of training set...")
predicted_testset_labels = knn.predict(testData)

# Evaluating KNN performance on TEST set
acc_score = accuracy_score(testLabels, predicted_testset_labels)
print(">>> Accuracy score for test set: {}".format(acc_score))
report = classification_report(testLabels, predicted_testset_labels)
print(">>> Classification report for test set: \n{}".format(report))
print("")
