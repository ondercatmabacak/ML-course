#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 04:39:59 2019

@author: ayf

Example:
-------
The 1st Classification Algorithm: K-Nearest Neighbor

KNN performance on toy datasets
"""

# import modules
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

# a method for closing all plot windows 
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
def plot_dataset(data, col1, col2, labels, title=""):
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
    for label in labels.unique():
        # get data points belonging to the group with 'label' labeled
        label_points = data[labels==label].values
        # get x and y points
        x_points = label_points[:,0]
        y_points = label_points[:,1]
        # determine label text in figure legend
        label_text = "class-{}".format(label)
        # scatter datapoints belonging to the 'label'
        plt.scatter(x_points, y_points, marker="o", alpha=0.70, label=label_text)
    
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

#%% create a classification model: KNN

# model hyper-parameters (values are up to you)
n_neigbors = 5 # how many neighbors will be checked during the prediction
minkowski_param = 2 # if p=1 then distance metric will be Manhattan, if p=2 than distance metric will be Euclidean

# create the model using the hyper parameters defined above
knn = KNeighborsClassifier(n_neighbors=n_neigbors, 
                           metric='minkowski', p=minkowski_param, 
                           n_jobs=3)

##################################################
################ TOY DATASETS  ###################
##################################################
#%% binary labeled blobs dataset
title =  "binary_blobs"
path = "../datasets/toy/{}.csv".format(title)

print("::::: Binary Labeled Blobs Dataset :::::\n")
data, labels = read_dataset(path, "label")
plot_dataset(data, "f0", "f1", labels, title)

knn.fit(data, labels)
preds = knn.predict(data)

plot_boundary(knn)

acc_score = accuracy_score(labels, preds)
print(">>> Accuracy Score: {}".format(acc_score))
conf_matrix = confusion_matrix(labels, preds)
print(">>> Confusion matrix: \n{}".format(conf_matrix))
report = classification_report(labels, preds)
print(">>> Classification Report: \n{}".format(report))
print("")



#%% binary labeled circles dataset
title =  "binary_circles"
path = "../datasets/toy/{}.csv".format(title)

print("::::: Binary Labeled Circles Dataset :::::\n")
data, labels = read_dataset(path, "label")
plot_dataset(data, "f0", "f1", labels, title)

knn.fit(data, labels)
preds = knn.predict(data)

plot_boundary(knn)

acc_score = accuracy_score(labels, preds)
print(">>> Accuracy Score: {}".format(acc_score))
conf_matrix = confusion_matrix(labels, preds)
print(">>> Confusion matrix: \n{}".format(conf_matrix))
report = classification_report(labels, preds)
print(">>> Classification Report: \n{}".format(report))
print("")

#%% binary labeled moons dataset
title =  "binary_moons"
path = "../datasets/toy/{}.csv".format(title)

print("::::: Binary Labeled Moons Dataset :::::\n")
data, labels = read_dataset(path, "label")
plot_dataset(data, "f0", "f1", labels, title)

knn.fit(data, labels)
preds = knn.predict(data)

plot_boundary(knn)

acc_score = accuracy_score(labels, preds)
print(">>> Accuracy Score: {}".format(acc_score))
conf_matrix = confusion_matrix(labels, preds)
print(">>> Confusion matrix: \n{}".format(conf_matrix))
report = classification_report(labels, preds)
print(">>> Classification Report: \n{}".format(report))
print("")

#%% binary labeled intermixed dataset
title =  "binary_intermixed"
path = "../datasets/toy/{}.csv".format(title)

print("::::: Binary Labeled Intermixed Dataset :::::\n")
data, labels = read_dataset(path, "label")
plot_dataset(data, "f0", "f1", labels, title)

knn.fit(data, labels)
preds = knn.predict(data)

plot_boundary(knn)

acc_score = accuracy_score(labels, preds)
print(">>> Accuracy Score: {}".format(acc_score))
conf_matrix = confusion_matrix(labels, preds)
print(">>> Confusion matrix: \n{}".format(conf_matrix))
report = classification_report(labels, preds)
print(">>> Classification Report: \n{}".format(report))
print("")

#%% multi labeled blobs dataset
title =  "multilabel_blobs"
path = "../datasets/toy/{}.csv".format(title)

print("::::: Multi Labeled Blobs Dataset :::::\n")
data, labels = read_dataset(path, "label")
plot_dataset(data, "f0", "f1", labels, title)

knn.fit(data, labels)
preds = knn.predict(data)

plot_boundary(knn)

acc_score = accuracy_score(labels, preds)
print(">>> Accuracy Score: {}".format(acc_score))
conf_matrix = confusion_matrix(labels, preds)
print(">>> Confusion matrix: \n{}".format(conf_matrix))
report = classification_report(labels, preds)
print(">>> Classification Report: \n{}".format(report))
print("")

#%% multi labeled intermixed dataset
title =  "multilabel_intermixed"
path = "../datasets/toy/{}.csv".format(title)

print("::::: Multi Labeled Intermixed Dataset :::::\n")
data, labels = read_dataset(path, "label")
plot_dataset(data, "f0", "f1", labels, title)

knn.fit(data, labels)
preds = knn.predict(data)

plot_boundary(knn)

acc_score = accuracy_score(labels, preds)
print(">>> Accuracy Score: {}".format(acc_score))
conf_matrix = confusion_matrix(labels, preds)
print(">>> Confusion matrix: \n{}".format(conf_matrix))
report = classification_report(labels, preds)
print(">>> Classification Report: \n{}".format(report))
print("")