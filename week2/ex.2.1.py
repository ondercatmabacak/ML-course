#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 05:59:21 2019

@author: ayf

Example:
-------
The 1st Classification Algorithm: K-Nearest Neighbor

Training model and predicting some novel instances
Displaying decision boundary
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
    for label in sorted(labels.unique()):
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


# method for displaying a single point
def disp_point(point):
   plt.scatter(point[:,0], point[:,1], marker="o", color="black", s=60, alpha=0.7)
   plt.text(point[0,0], point[0,1], "?", fontsize=15, 
            verticalalignment="bottom",
            horizontalalignment="left")

#%% create a classification model: KNN

# model hyper-parameters (values are up to you)
n_neigbors = 3 # how many neighbors will be checked during the prediction
minkowski_param = 2 # if p=1 then distance metric will be Manhattan, if p=2 than distance metric will be Euclidean

# create the model using the hyper parameters defined above
knn = KNeighborsClassifier(n_neighbors=n_neigbors, 
                           metric='minkowski', p=minkowski_param, 
                           n_jobs=3)
# n_jobs -> how many threads for parallel computing ? (it can be equal to # of cores - 1)

#%% load blobs dataset and visualize
title =  "binary_blobs"
path = "../datasets/toy/{}.csv".format(title)

data, labels = read_dataset(path, "label")
# display the scatter plot for this data set
plot_dataset(data, "f0", "f1", labels, title) 

#%% 1st step: train (fit) the model with dataset
print(">>> Model is learning the data...")
knn.fit(data, labels)
print(">>> Finished.")

#%% predict the label of a single data point
# determine a single point for testing
point = np.array([[-2, 4]])
# display the point 
disp_point(point)
# what should the label of that point be ? predict yourself.
# what KNN predicts ?
predicted_label = knn.predict(point)
print(">>> KNN says the label of {} is {}".format(point, predicted_label))
 
#%% predict the label of single data point

# determine a single point for testing
point = np.array([[-6, 4]])
# display the point 
disp_point(point)
# what should the label of that point be ? predict yourself.
# what KNN predicts ?
predicted_label = knn.predict(np.array(point))
print(">>> KNN says the label of {} is {}".format(point, predicted_label))

#%% label of grid points ?
plot_dataset(data, "f0", "f1", labels, title) 
# get min and max values for each feature
mins = data.min()
maxs = data.max()
# create a grid 
vec1 = np.arange(np.floor(mins["f0"]), np.ceil(maxs["f0"]), step=0.5)
vec2 = np.arange(np.floor(mins["f1"]), np.ceil(maxs["f1"]), step=0.5)
gridX, gridY = np.meshgrid(vec1, vec2) 
grid = np.c_[gridX.ravel(), gridY.ravel()]
## what is the labels of each grid point
for point in grid:
    disp_point(point.reshape(1,-1))
    
#%% predict label of grid points
plot_dataset(data, "f0", "f1", labels, title)    
# predict the labels of grid points using knn
preds = knn.predict(grid)

### now, plot the grid with predicted labels
# scatter grid points for predicted label 1
x_points = grid[preds==0,0]
y_points = grid[preds==0,1]
plt.scatter(x_points,y_points, marker="o", alpha=0.60, edgecolor="tab:blue", 
            facecolor="none", label="predicted class: 0", zorder=-10)
# scatter grid points for predicted label 1
x_points = grid[preds==1,0]
y_points = grid[preds==1,1]
plt.scatter(x_points,y_points, marker="o", alpha=0.60, edgecolor="tab:orange", 
            facecolor="none", label="predicted class: 1", zorder=-10)

#%% display decision boundary 

h = .01  # step size in the mesh grid (if h is too low, then there might be memory error !)
# determine bounds of meshgrid 
_x_min, _x_max = mins["f0"], maxs["f0"]
_y_min, _y_max = mins["f1"], maxs["f1"]
# extend bounds about amount of 5%
x_min = _x_min - np.abs(_x_max-_x_min)*0.05
x_max = _x_max + np.abs(_x_max-_x_min)*0.05
y_min = _y_min - np.abs(_y_max-_y_min)*0.05
y_max = _y_max + np.abs(_y_max-_y_min)*0.05
# generate a mesh grid
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# convert mesh grid to vector
grid_points = np.c_[xx.ravel(), yy.ravel()]
# predict the label of each data points in grid points
Z = knn.predict(grid_points)
# reshape Z so it will be a grid again
Z = Z.reshape(xx.shape)
# plot grid points with their predicted labels (by KNN) that corresponds to a color
plt.imshow(Z, interpolation="gaussian", zorder=-100, 
           alpha=0.3, extent=[x_min, x_max, y_min, y_max], 
           aspect="auto", origin="lower")
