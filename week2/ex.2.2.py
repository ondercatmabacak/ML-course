#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 07:33:49 2019

@author: ayf

Example:
-------
The 1st Classification Algorithm: K-Nearest Neighbor

Understanding effects of hyper-parameters on model
"""

# import modules
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


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

# method for displaying decision boundary
def plot_boundary(model, data, col0, col1):
    # get bounds of data
    mins = data[[col0, col1]].min()
    maxs = data[[col0, col1]].max()
    # determine bounds of meshgrid 
    xmin, xmax = mins[0], maxs[0]
    ymin, ymax = mins[1], maxs[1]
    
    # extend bounds about amount of 5%
    x_min = xmin - np.abs(xmax-xmin)*0.05
    x_max = xmax + np.abs(xmax-xmin)*0.05
    y_min = ymin - np.abs(ymax-ymin)*0.05
    y_max = ymax + np.abs(ymax-ymin)*0.05
    
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


# method for displaying a single point
def disp_point(ax, point):
   ax.scatter(point[:,0], point[:,1], marker="o", color="black", s=60, alpha=0.7)
   ax.text(point[0,0], point[0,1], "?", fontsize=15, 
            verticalalignment="bottom",
            horizontalalignment="left")
   
#%% create two KNN models with different hyper-parameter set

# KNN with n_neighbors=5 and Euclidean distance metric
knn1 = KNeighborsClassifier(n_neighbors=3, 
                           metric='minkowski', p=2, 
                           n_jobs=3)
# KNN with n_neighbors=10 and Manhattan distance metric
knn2 = KNeighborsClassifier(n_neighbors=10, 
                           metric='minkowski', p=1, 
                           n_jobs=3)

#%% load intermixed dataset
title =  "binary_intermixed"
path = "../datasets/toy/{}.csv".format(title)

print("::::: Binary Labeled Intermixed Dataset :::::\n")
data, labels = read_dataset(path, "label")

#%% Testing knn1 on dataset

# display the scatter plot for this data set
plot_dataset(data, "f0", "f1", labels, "KNN\n(n_neighbors=5 & Euclidean distance)") 
# train (fit) the model with dataset
print(">>> 'knn1' is learning the data...")
knn1.fit(data, labels)
# display boundaries
plot_boundary(knn1, data, "f0", "f1")

# get ax from plt
ax1 = plt.gca()

#%% Testing knn2 on dataset

# display the scatter plot for this data set
plot_dataset(data, "f0", "f1", labels, "KNN\n(n_neighbors=10 & Manhattan distance)") 
# train (fit) the model with dataset
print(">>> 'knn2' is learning the data...")
knn2.fit(data, labels)
# display boundaries
plot_boundary(knn2, data, "f0", "f1")

# get ax from plt
ax2 = plt.gca()

#%% predicting the label of a novel instance
# determine a single point for testing
point = np.array([[-0.5, 0.0]])

# display the point on ax1 (1st plot)
disp_point(ax1, point)
# what should the label of that point be according to knn1 ?
prediction_knn1 = knn1.predict(np.array(point))
print(">>> knn1 says the label of {} is {}".format(point, prediction_knn1))

# display the point on ax2 (2nd plot)
disp_point(ax2, point)
# what should the label of that point be according to knn2 ?
prediction_knn2 = knn2.predict(np.array(point))
print(">>> knn2 says the label of {} is {}".format(point, prediction_knn2))

# why predictions of knn1 and knn2 are different ?
# which one should we trust on ? 
# on what basis ?