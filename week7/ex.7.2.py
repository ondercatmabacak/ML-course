#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 09:11:37 2019

@author: ayf

GMM: How it works ?
"""

# import modules
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture

   
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
                   alpha=0.8, zorder=100)


# method for displaying center point of cluster
def disp_cluster_ellipses(means, covs, factor=1.0):
    ax = plt.gca()
    # for each cluster
    for mean, covar in zip(means, covs):
         # calculate eigenvectors and eigenvalues
        eig_vec,eig_val,_ = np.linalg.svd(covar)
        # Make sure 0th eigenvector has positive x-coordinate
        if eig_vec[0][0] < 0:
            eig_vec[0] *= -1
        # calculate major and minor axes length
        majorLength = 2*factor*np.sqrt(eig_val[0])
        minorLength = 2*factor*np.sqrt(eig_val[1])
        # calculate rotation angle
        u = eig_vec[0] / np.linalg.norm(eig_vec[0])
        angle = np.arctan(u[1]/u[0])
        angle = 180.0*angle/np.pi
        
        # create ellipse
        ell = mpl.patches.Ellipse(mean, majorLength, minorLength, angle, 
                                  linestyle="-", linewidth=2, 
                                  edgecolor="black", facecolor='none', 
                                  alpha=0.85, zorder=-100)
        ell.set_clip_box(ax.bbox)
        ax.add_artist(ell)



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
max_iter   = 20

# initializing means of clusters
center_0 = [-3, -2]
center_1 = [-3, -3]
cluster_means = np.array([center_0, center_1])

# initialize covariance of clusters
cov_0 = [[1.0, 0], 
         [0, 2.0]]
cov_1 = [[1.5, 0.5], 
         [0.5, 1.5]]
cluster_covs = np.array([cov_0, cov_1])
cluster_covs_inv = np.linalg.inv(cluster_covs)

# initialize weights of clusters
weight_0 = 0.25
weight_1 = 0.75
cluster_weights = np.array([weight_0, weight_1])


#%% start EM algorithm

for iter_count in range(1, max_iter):
    
    print("====================")
    print(">>> Iteration: {}".format(iter_count))
    
    # reset plot interface
    plt.clf()
    plt.gca().set_prop_cycle(None) 
    
    # display the scatter plot for this data set (without labels)
    plot_dataset(data, "f0", "f1", labels=None, title=iter_count)
    
    # display cluster centers
    disp_cluster_centers(cluster_means)
    # display cluster ellipses
    disp_cluster_ellipses(cluster_means, cluster_covs)
    
    print("  > Current State")
    for i, (m, c, w) in enumerate(zip(cluster_means, cluster_covs, cluster_weights)):
        print("  > Cluster-{} params:".format(i))
        print("    Mean: {}".format(m))
        print("    Cov: \n{}".format(c))
        print("    Weight: {}".format(w))
    pause()
    
    #create GMM initialized with hyper params
    gmm = GaussianMixture(n_components=n_clusters, n_init=1, max_iter=iter_count,
                          means_init=cluster_means, precisions_init=cluster_covs_inv,
                          weights_init=cluster_weights, random_state=22)
 
    ##############
    # trick: gmm is NOT fitted yet, but we fake the model (do not try at home)
    gmm.weights_ = cluster_weights
    gmm.means_ = cluster_means
    from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
    gmm.precisions_cholesky_ = _compute_precision_cholesky(cluster_covs, gmm.covariance_type)
    gmm.means_ = cluster_means
    ##############
    
    # get labels of data
    labels = gmm.predict(data)
    
    # reset plot interface
    plt.clf()
    plt.gca().set_prop_cycle(None) 
    
    # display data set (without labels) 
    plot_dataset(data, "f0", "f1", labels=labels, fill=False, 
                 title="iteration: {}".format(iter_count))
    
    # display cluster centers
    disp_cluster_centers(cluster_means)
    # display cluster ellipses
    disp_cluster_ellipses(cluster_means, cluster_covs)
    
    print("  > Expectation Step completed")
    pause()
    
    # train model
    gmm.fit(data)
    
    # get updated means
    cluster_means = gmm.means_
    # get updated covs
    cluster_covs = gmm.covariances_
    cluster_covs_inv = gmm.precisions_
    # get updated weights
    cluster_weights = gmm.weights_
    
    # reset plot interface
    plt.clf()
    plt.gca().set_prop_cycle(None) 
    
    # display data set (without labels) 
    plot_dataset(data, "f0", "f1", labels=labels, fill=False, 
                 title="iteration: {}".format(iter_count))
    
    # display cluster centers
    disp_cluster_centers(cluster_means)
    # display cluster ellipses
    disp_cluster_ellipses(cluster_means, cluster_covs)
    
    print("  > Maximiization Step completed")
    for i, (m, c, w) in enumerate(zip(cluster_means, cluster_covs, cluster_weights)):
        print("  > Cluster-{} params:".format(i))
        print("    Mean: {}".format(m))
        print("    Cov: \n{}".format(c))
        print("    Weight: {}".format(w))
    pause()
    