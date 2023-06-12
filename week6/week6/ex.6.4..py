#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:31:15 2019

@author: ayf

1st Clustering Algorithm: K-Means

Examining model hyper-parameters (max-iter)
"""


# import modules
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


   
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
                   alpha=0.7, zorder=100)

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
    # plot grid points with their predicted labels that corresponds to a color
    plt.imshow(Z, interpolation="gaussian", zorder=-100, 
               alpha=0.2, extent=[x_min, x_max, y_min, y_max], 
               aspect="auto", origin="lower")

# switch data labels
def switch_labels(arr, label_map):
    # copy input list
    inp = np.copy(arr)
    
    import time
    # method for getting current milliseconds time (for obtaining unique numbers)
    current_time_ms = lambda: int(round(time.time() * 1000))
    
    # how many unique labels
    n = len(label_map)
    # generate intermediate labels (must be unique)
    tmp = current_time_ms() % 1000
    intermediates = [tmp + i for i in xrange(n)]
    
    # create two maps (from current to intermediate && from intermediate to target value)
    map1, map2 = {}, {}
    for current_label, intermediate_label in zip(label_map.keys(), intermediates): 
        map1[current_label] = intermediate_label
        map2[intermediate_label] = label_map[current_label]
    
    # 1st iteration, switch current labels with intermediate values
    for current_label, intermediate_label in map1.iteritems():
        inp[inp==current_label] = intermediate_label
    # 2nd iteration, switch intermediate labels with target labels
    for intermediate_label, target_label in map2.iteritems():
        inp[inp==intermediate_label] = target_label
    
    # return array
    return inp
    
# method for asking user input (for blocking current operation)
def pause():
    plt.draw()
    plt.pause(0.20)
    raw_input("Press enter to continue ...")


#%% load toy dataset and visualize
title =  "binary_intermixed"
path = "../datasets/toy/{}.csv".format(title)

data, true_labels = read_dataset(path, "label")
# display the scatter plot for this data set (with labels)
plt.figure()
plot_dataset(data, "f0", "f1", true_labels, title=title+" (supervised)")
# display the scatter plot for this data set (without labels)
plt.figure()
plot_dataset(data, "f0", "f1", labels=None, title=title+" (unsupervised)")


#%% create a clustering model: K-Means

# model hyper-parameters (values are up to you)
n_clusters = 2 
max_iters  = range(1, 10)

# initializing means of clusters
center_0 = [-3, -2]
center_1 = [-3, -3]
default_cluster_centers = np.array([center_0, center_1])

#################
for max_iter in max_iters:
    # pause the code
    pause()
    
    print("====================")
    print(">>> Max-Iteration: {}".format(max_iter))
    
    # create the model 
    kms = KMeans(n_clusters=n_clusters, n_init=1, init=default_cluster_centers,
                 max_iter=max_iter, tol=1e-9, random_state=22, n_jobs=3)
    # n_jobs -> how many threads for parallel computing ? (it can be equal to # of cores - 1)
    
    # train (fit) the model with data but without labels !!!
    print("  > Model is learning the data...")
    kms.fit(data)
    print("  > Finished.")
    
    cluster_centers = kms.cluster_centers_
    labels = kms.labels_
    
    # reset plot interface
    plt.clf()
    plt.gca().set_prop_cycle(None) 
    
    # display data set 
    plot_dataset(data, "f0", "f1", labels=labels, fill=False, 
                 title="Max-Iteration: {}".format(max_iter))
    # display cluster centers
    disp_cluster_centers(cluster_centers)
    
    # calculate the model performance
    sil_avg = silhouette_score(data, labels)
    print("  > AVG Silhouette Score: {}".format(sil_avg))
    
   