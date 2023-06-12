#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:06:07 2019

@author: ayf

1st Clustering Algorithm: K-Means

How to determine number of clusters ?
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
    
    # cmap = list(plt.get_cmap("tab10").colors)
    cmap = plt.get_cmap("tab10")
    
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
                # color_index = label%len(cmap)
                color = cmap(label)
                plt.scatter(x_points, y_points, marker="o", facecolor="none",
                            edgecolor=color, linewidths=2.0,
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
title =  "clusters1"
path = "../datasets/toy/{}.csv".format(title)

data, true_labels = read_dataset(path, "label")
# display the scatter plot for this data set (without labels)
plt.figure()
plot_dataset(data, "f0", "f1", labels=None, title=title+" (unsupervised)")


#%% create a clustering model: K-Means

# model hyper-parameters (values are up to you)
max_iter  = 100
n_clusterss = range(2, 12) 

# dict variable to keep created models and scores 
result = {}
#################
for n_clusters in n_clusterss:
    # pause the code
    # pause()
    
    print("====================")
    print(">>> Number of Clusters: {}".format(n_clusters))
    
    # create the model 
    kms = KMeans(n_clusters=n_clusters, n_init=1, max_iter=max_iter, 
                 tol=1e-9, random_state=22, n_jobs=3)
    # n_jobs -> how many threads for parallel computing ? (it can be equal to # of cores - 1)
    
    # train (fit) the model with data but without labels !!!
    kms.fit(data)
    
    cluster_centers = kms.cluster_centers_
    labels = kms.labels_
    
    # reset plot interface
    plt.clf()
    plt.gca().set_prop_cycle(None) 
    
    # display data set 
    plot_dataset(data, "f0", "f1", labels=labels, fill=False, 
                 title="N Clusters: {}".format(n_clusters))
    # display cluster centers
    disp_cluster_centers(cluster_centers)
    
    # model performance
    score = kms.score(data, labels)
    print("  > Score: {}".format(score))
    sil_avg = silhouette_score(data, labels)
    print("  > AVG Silhouette Score: {}".format(sil_avg))
    
    # keep the scores
    result[n_clusters] = {"model": kms,
                          "score": score, 
                          "sil_score": sil_avg}
    

#%% print scores with respect to "number of clusters"

print("")
print(">>> Scores: ")
for n_cluster in result.keys():
    score = result[n_cluster]["score"]
    sil_avg = result[n_cluster]["sil_score"]
    print ("  > n_cluster:  {};  score: {:.2f}  silhouette score: {: .2f}"
           .format(n_cluster, score, sil_avg))


#%% plot the scores against number of clusters

# get number of cluster list
n_cluster_list = result.keys()
# get silhouette score list
sil_list = [result[n]["sil_score"] for n in result.keys()]
# get score list
score_list = [result[n]["score"] for n in result.keys()]

# create figure
plt.figure()
# get axis
ax1 = plt.gca()
# create second axis
ax2 = ax1.twinx()

# plot the silhouette score 
ax1.scatter(n_cluster_list, sil_list, color="tab:blue", label="silhouette score")
ax1.plot(n_cluster_list, sil_list, color="tab:blue")
# plot the score
ax2.scatter(n_cluster_list, score_list, color="tab:red", label="score")
ax2.plot(n_cluster_list, score_list, color="tab:red")

# edit labels and other staffs
ax1.set_xlabel("n_clusters", size=14)
ax1.set_ylabel("silhouette score", size=14)
ax1.legend(loc="best")
ax1.set_xticks(n_cluster_list)
ax1.legend(loc='upper left')
ax1.grid(axis="x")

ax2.set_ylabel("score", size=14)
ax2.legend(loc='upper right')
 

#%% determining best value for n_clusters

# highest silhouette score is better
best_n_clusters = None
max_sil_score   = -np.inf

for n_clusters in result.keys():
    if result[n_clusters]["sil_score"] > max_sil_score:
        max_sil_score = result[n_clusters]["sil_score"]
        best_n_clusters = n_clusters

print(">>> Max Silhouette Score    : {:.2f}".format(max_sil_score))
print(">>> Best Number of Clusters : {}".format(best_n_clusters))
best_n_clusters = 6
# get best model
kms_best = result[best_n_clusters]["model"]
# display clustering result of best model
labels = kms_best.labels_
plt.figure()
plot_dataset(data, "f0", "f1", labels, title="N Clusters: {}".format(best_n_clusters))


#%% display the data with true labels
# display the scatter plot for this data set (with labels)
plt.figure()
plot_dataset(data, "f0", "f1", true_labels, title=title+" (supervised)")
 