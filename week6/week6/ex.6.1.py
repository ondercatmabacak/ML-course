#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 00:25:51 2019

@author: ayf

1st Clustering Algorithm: K-Means
"""

# import modules
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

   
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
def plot_dataset(data, col1, col2, labels=None, fill=True, title=""):
    '''
    type of data and labels parameters are "pandas frame" and "pandas series"
    col1 and col2 are the features which will be scattered 
    if labels are specified, use different color for each cluster set
    '''
    # first, get the corresponding columns from the dataset
    data = data.loc[:, [col1, col2]] # get all rows for only col1 and col2
    
    # create a figure
    plt.figure()
    plt.title(title, size=20)
    
    cmap = list(plt.get_cmap("tab10").colors)
    
    # scatter each unique group with a different color
    if labels is not None: 
        for label in sorted(np.unique(labels)):
            # get data points belonging to the group with 'label' labeled
            label_points = data[labels==label].values
            # get x and y points
            x_points = label_points[:,0]
            y_points = label_points[:,1]
            # determine label text in figure legend
            label_text = "class-{}".format(label)
            # scatter datapoints belonging to the 'label'
            if fill:
                plt.scatter(x_points, y_points, marker="o", alpha=0.70, label=label_text)
            else:
                plt.scatter(x_points, y_points, marker="o", facecolor="none",
                            edgecolor=cmap.pop(), linewidths=2.0,
                            alpha=0.90, label=label_text)
                
        # place legend box at the best location
        plt.legend(loc="best", fontsize=14)
        # display labels for x and y axes
                
    else:
        # get x and y points
        x_points = data.values[:,0]
        y_points = data.values[:,1]
        # scatter datapoints belonging to the 'label'
        plt.scatter(x_points, y_points, marker="o", color="gray", alpha=0.70)
    
    plt.xlabel(col1, size=14)
    plt.ylabel(col2, rotation=0, size=14)

def disp_point_label(point, label):
    plt.text(point[0], point[1], label, fontsize=15, 
            verticalalignment="bottom",
            horizontalalignment="left")


#%% load blobs dataset and visualize
title =  "binary_blobs"
path = "../datasets/toy/{}.csv".format(title)

data, labels = read_dataset(path, "label")
# display the scatter plot for this data set (with labels)
plot_dataset(data, "f0", "f1", labels, title=title+" (supervised)")
# display the scatter plot for this data set (without labels)
plot_dataset(data, "f0", "f1", labels=None, title=title+" (unsupervised)")


#%% create a clustering model: K-Means

# model hyper-parameters (values are up to you)
n_clusters = 2 
n_init     = 1
max_iter   = 100 
tol        = 1e-4
  
# create the model using the hyper parameters defined above
kms = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter,
             tol=tol, random_state=22, n_jobs=3)
# n_jobs -> how many threads for parallel computing ? (it can be equal to # of cores - 1)


#%% train (fit) the model with data but without labels !!!
print(">>> Model is learning the data...")
kms.fit(data)
print(">>> Finished.")
# what did the model learned from unlabeled data ?

#%% examine trained model

#### get labels of train data
# labels = kms.predict(data) ==> not necessary !!!
#!!! you do not have to predict labels of data instances that model fitted !!!
#!!! labels are assigned during the training procedure
labels = kms.labels_
# how many cluster found ? (it should be equal to 'n_clusters' hyper-parameter)
clusters = np.unique(labels)
print( ">>> Discovered clusters: {}".format(clusters))
# display label of each data point
for point, label in zip(data.values, labels):
    disp_point_label(point, label)
 
#%% display decision boundary of "clusters"

x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()

# set step size
h = .05  # step size in the mesh grid (if h is too low, then there might be memory error !)
# generate a mesh grid
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# convert mesh grid to vector
grid_points = np.c_[xx.ravel(), yy.ravel()]
# predict the label of each data points in grid points
Z = kms.predict(grid_points)
# reshape Z so it will be a grid again
Z = Z.reshape(xx.shape)
# plot grid points with their predicted labels that corresponds to a color
plt.imshow(Z, interpolation="gaussian", zorder=-100, 
           alpha=0.2, extent=[x_min, x_max, y_min, y_max], 
           aspect="auto", origin="lower")

