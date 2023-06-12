#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 00:54:24 2019

@author: ayf

1st Clustering Algorithm: K-Means 

How it works ?
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
    #plt.figure()
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

# method for displaying center point of cluster
def disp_cluster_centers(centers):
   for cid, center in enumerate(centers):
       plt.text(center[0], center[1], cid, fontsize=15, 
            verticalalignment="bottom",
            horizontalalignment="left")
       plt.scatter(centers[cid,0], centers[cid,1], marker="*", color="black", s=60, 
                   alpha=0.8, zorder=100)

# method for asking user input (for blocking current operation)
def pause():
    plt.draw()
    plt.pause(0.20)
    raw_input("Press enter to continue ...")


#%% load toy dataset and visualize
title =  "binary_intermixed"
path = "../datasets/toy/{}.csv".format(title)

data, labels = read_dataset(path, "label")
# display the scatter plot for this data set (with labels)
plt.figure()
plot_dataset(data, "f0", "f1", labels, title=title+" (supervised)")
# display the scatter plot for this data set (without labels)
plt.figure()
plot_dataset(data, "f0", "f1", labels=None, title=title+" (unsupervised)")


    
#%% how K-Means works ?  

# model hyper-parameters (values are up to you)
n_clusters = 2 
max_iter   = 10

# initializing means of clusters
center_0 = [-3, -2]
center_1 = [-3, -3]
cluster_centers = np.array([center_0, center_1])

# display the scatter plot for this data set (without labels)
plt.figure()
plot_dataset(data, "f0", "f1", labels=None, title=title+" (unsupervised)")


#################
for iter_count in range(max_iter):
    print("====================")
    print(">>> Iteration: {}".format(iter_count))
    
    # reset plot interface
    plt.clf()
    plt.gca().set_prop_cycle(None) 
    
    # display data set (without labels) 
    plot_dataset(data, "f0", "f1", labels=None, 
                 title="iteration: {}".format(iter_count))
    # display cluster centers
    disp_cluster_centers(cluster_centers)
    
    # pause the code
    pause()
    
    print("  > calculating distances ...")
    # distance calculations
    distances = np.zeros(shape=(data.shape[0], 0))
    # for each cluster center
    for cluster_id, center in enumerate(cluster_centers):
        # calculate Euclidean distance between cluster center and each data instances
        dists = [np.linalg.norm(center-_x) for _x in data.values]
        dists = np.array(dists).reshape(-1,1)
        # append distances to the list
        distances = np.hstack((distances,dists))
    
    print("  > determining data labels  ...")
    # determine cluster labels for each data instances
    labels = []
    for d in distances:
        # get minimum distance for this instance
        min_dist = np.min(d)
        # find the label of clusest cluster
        label  = np.where(d==min_dist)[0][0]
        # add label to the list
        labels.append(label)
    # convert list to numpy array
    labels = np.array(labels)
    print("!!! data labels are updated ...")
    
    # reset plot interface
    plt.clf()
    plt.gca().set_prop_cycle(None) 
    
    # display data set with labels
    plot_dataset(data, "f0", "f1", labels=labels, fill=False,
                 title="iteration: {}".format(iter_count))
    # display cluster centers
    disp_cluster_centers(cluster_centers)
    
    # pause the code
    pause()
    
    # update cluster centers
    print("  > calculating new cluster centers ...")
    cluster_centers = []
    for label in np.unique(labels):
        # get data instances with current label
        subdata = data.loc[labels==label, :]
        # get mean of the each dimension (feature)
        center = subdata.mean().values.tolist()
        cluster_centers.append(center)
    # convert data type
    cluster_centers = np.array(cluster_centers)
    print("!!! cluster centers are updated ...")
   
    # reset plot interface
    plt.clf()
    plt.gca().set_prop_cycle(None) 
        
    # display data set with labels
    plot_dataset(data, "f0", "f1", labels=labels, fill=False,
                 title="iteration: {}".format(iter_count))
    # display cluster centers
    disp_cluster_centers(cluster_centers)
    
    # pause the code
    pause()
   