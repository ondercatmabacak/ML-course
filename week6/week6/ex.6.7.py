#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:25:58 2019

@author: ayf

1st Clustering Algorithm: K-Means

Model Performance on Toy Dataset ?
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


def disp_scores(result):
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

def estimate_n_clusters(data, n_cluster_set):
    # dict variable to keep created models and scores 
    result = {}
    #################
    for n_clusters in n_cluster_set:
        # create the model 
        kms = KMeans(n_clusters=n_clusters, n_init=100, max_iter=1000, 
                     tol=1e-3, random_state=22, n_jobs=3)
        # train (fit) the model with data but without labels !!!
        kms.fit(data)
        # get labels
        labels = kms.labels_
        # model performance
        score = kms.score(data)
        sil_avg = silhouette_score(data, labels)
        # print scores
        print ("  > n_cluster:  {};  score: {:.2f}  silhouette score: {: .2f}"
           .format(n_clusters, score, sil_avg))
        # keep the scores
        result[n_clusters] = {"model": kms,"score": score, "sil_score": sil_avg}
    
    #display scores
    disp_scores(result)
    
    ### determine best model
    # highest silhouette score is better
    best_n_clusters = None
    max_sil_score   = -np.inf
    
    for n_clusters in result.keys():
        if result[n_clusters]["sil_score"] > max_sil_score:
            max_sil_score = result[n_clusters]["sil_score"]
            best_n_clusters = n_clusters
    
    print("  > Max Silhouette Score    : {:.2f}".format(max_sil_score))
    print("  > Best Number of Clusters : {}".format(best_n_clusters))
    # get best model
    kms_best = result[best_n_clusters]["model"]
    return kms_best

# method for testing model on a dataset
def test_model(data, n_clusters):
    print ("====================")
    print (">>> Testing Model")
    print ("  > n_clusters: {}".format(n_clusters))
    
    # if number of clusters type is list, then estimate the best value
    if isinstance(n_clusters, list):
        model = estimate_n_clusters(data, n_cluster_set=n_clusters)
        n_clusters = model.n_clusters
        title = "N Clusters: {} (estimated)".format(n_clusters)
    elif isinstance(n_clusters, int):
        model = KMeans(n_clusters=n_clusters, n_init=100, max_iter=1000, 
                     tol=1e-3, random_state=22, n_jobs=3)
        model.fit(data)
        title = "N Clusters: {}".format(n_clusters)
    else:
        raise Exception("invalid n_clusters")
    
    # get labels of data instances
    labels = model.labels_
    # get cluster centers
    cluster_centers = model.cluster_centers_
    
    # plot clusters with boundary
    plt.figure()
    plot_dataset(data, "f0", "f1", labels=labels, fill=False,
                 title=title)
    disp_cluster_centers(cluster_centers)
    plot_boundary(model)
 
    # evaluate model performance (unsupervised)
    score = silhouette_score(data, labels)
    print("  > Silhouette Score: {}".format(score))
    
##################################################
################ TOY DATASETS  ###################
##################################################
    
#%% binary labeled blobs dataset
closeAll()
# read data set
dataset_name = "binary_blobs"
path = "../datasets/toy/{}.csv".format(dataset_name)
print("\n::::: {} :::::\n".format(dataset_name))
data, true_labels = read_dataset(path, "label")

# display the scatter plot for this data set (with labels)
plt.figure()
plot_dataset(data, "f0", "f1", true_labels, title=dataset_name+" (supervised)")
# display the scatter plot for this data set (without labels)
plt.figure()
plot_dataset(data, "f0", "f1", labels=None, title=dataset_name+" (unsupervised)")


# test model (estimate number of clusters)
test_model(data, n_clusters=range(2, 6))

#%% binary labeled circles dataset
closeAll()
dataset_name = "binary_circles"
path = "../datasets/toy/{}.csv".format(dataset_name)
print("\n::::: {} :::::\n".format(dataset_name))
data, true_labels = read_dataset(path, "label")

# display the scatter plot for this data set (with labels)
plt.figure()
plot_dataset(data, "f0", "f1", true_labels, title=dataset_name+" (supervised)")
# display the scatter plot for this data set (without labels)
plt.figure()
plot_dataset(data, "f0", "f1", labels=None, title=dataset_name+" (unsupervised)")

# test model (estimate number of clusters)
test_model(data, n_clusters=range(2, 6))
# test model (estimate number of clusters)
test_model(data, n_clusters=2)

#%% binary labeled moons dataset
closeAll()
dataset_name = "binary_moons"
path = "../datasets/toy/{}.csv".format(dataset_name)
print("\n::::: {} :::::\n".format(dataset_name))
data, true_labels = read_dataset(path, "label")

# display the scatter plot for this data set (with labels)
plt.figure()
plot_dataset(data, "f0", "f1", true_labels, title=dataset_name+" (supervised)")
# display the scatter plot for this data set (without labels)
plt.figure()
plot_dataset(data, "f0", "f1", labels=None, title=dataset_name+" (unsupervised)")

# test model (estimate number of clusters)
test_model(data, n_clusters=range(2, 6))
# test model (estimate number of clusters)
test_model(data, n_clusters=5)

#%% binary labeled intermixed dataset
closeAll()
dataset_name = "binary_intermixed"
path = "../datasets/toy/{}.csv".format(dataset_name)
print("\n::::: {} :::::\n".format(dataset_name))
data, true_labels = read_dataset(path, "label")

# display the scatter plot for this data set (with labels)
plt.figure()
plot_dataset(data, "f0", "f1", true_labels, title=dataset_name+" (supervised)")
# display the scatter plot for this data set (without labels)
plt.figure()
plot_dataset(data, "f0", "f1", labels=None, title=dataset_name+" (unsupervised)")

# test model (estimate number of clusters)
test_model(data, n_clusters=range(2, 6))
# test model (estimate number of clusters)
test_model(data, n_clusters=2)

#%% multi labeled blobs dataset
closeAll()
dataset_name = "multilabel_blobs"
path = "../datasets/toy/{}.csv".format(dataset_name)
print("\n::::: {} :::::\n".format(dataset_name))
data, true_labels = read_dataset(path, "label")

# display the scatter plot for this data set (with labels)
plt.figure()
plot_dataset(data, "f0", "f1", true_labels, title=dataset_name+" (supervised)")
# display the scatter plot for this data set (without labels)
plt.figure()
plot_dataset(data, "f0", "f1", labels=None, title=dataset_name+" (unsupervised)")

# test model (estimate number of clusters)
test_model(data, n_clusters=range(2, 6))
 

#%% multi labeled intermixed dataset
closeAll()
dataset_name = "multilabel_intermixed"
path = "../datasets/toy/{}.csv".format(dataset_name)
print("\n::::: {} :::::\n".format(dataset_name))
data, true_labels = read_dataset(path, "label")

# display the scatter plot for this data set (with labels)
plt.figure()
plot_dataset(data, "f0", "f1", true_labels, title=dataset_name+" (supervised)")
# display the scatter plot for this data set (without labels)
plt.figure()
plot_dataset(data, "f0", "f1", labels=None, title=dataset_name+" (unsupervised)")

# test model (estimate number of clusters)
test_model(data, n_clusters=range(2, 6))
# test model (estimate number of clusters)
test_model(data, n_clusters=3)

#%% clusters1 dataset
closeAll()
dataset_name = "clusters1"
path = "../datasets/toy/{}.csv".format(dataset_name)
print("\n::::: {} :::::\n".format(dataset_name))
data, true_labels = read_dataset(path, "label")

# display the scatter plot for this data set (with labels)
plt.figure()
plot_dataset(data, "f0", "f1", true_labels, title=dataset_name+" (supervised)")
# display the scatter plot for this data set (without labels)
plt.figure()
plot_dataset(data, "f0", "f1", labels=None, title=dataset_name+" (unsupervised)")

# test model (estimate number of clusters)
test_model(data, n_clusters=range(2, 10))
# test model (estimate number of clusters)
test_model(data, n_clusters=3)

#%% clusters2 dataset
closeAll()
dataset_name = "clusters2"
path = "../datasets/toy/{}.csv".format(dataset_name)
print("\n::::: {} :::::\n".format(dataset_name))
data, true_labels = read_dataset(path, "label")

# display the scatter plot for this data set (with labels)
plt.figure()
plot_dataset(data, "f0", "f1", true_labels, title=dataset_name+" (supervised)")
# display the scatter plot for this data set (without labels)
plt.figure()
plot_dataset(data, "f0", "f1", labels=None, title=dataset_name+" (unsupervised)")

# test model (estimate number of clusters)
test_model(data, n_clusters=range(2, 10))
# test model (estimate number of clusters)
test_model(data, n_clusters=6)

#%% clusters3 dataset
closeAll()
dataset_name = "clusters3"
path = "../datasets/toy/{}.csv".format(dataset_name)
print("\n::::: {} :::::\n".format(dataset_name))
data, true_labels = read_dataset(path, "label")

# display the scatter plot for this data set (with labels)
plt.figure()
plot_dataset(data, "f0", "f1", true_labels, title=dataset_name+" (supervised)")
# display the scatter plot for this data set (without labels)
plt.figure()
plot_dataset(data, "f0", "f1", labels=None, title=dataset_name+" (unsupervised)")

# test model (estimate number of clusters)
test_model(data, n_clusters=range(2, 10))
# test model (estimate number of clusters)
test_model(data, n_clusters=5)
