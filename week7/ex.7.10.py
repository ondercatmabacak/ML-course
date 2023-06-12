#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 17:20:34 2019

@author: ayf

GMM variations: BGMM
"""

# import modules
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture
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
def disp_cluster_means(centers):
   for cid, center in enumerate(centers):
       plt.text(center[0], center[1], cid, fontsize=15, 
            verticalalignment="bottom",
            horizontalalignment="left")
       plt.scatter(centers[cid,0], centers[cid,1], marker="*", color="black", s=60, 
                   alpha=0.8, zorder=100)


# method for displaying center point of cluster
def disp_cluster_ellipses(means, covs, cov_type, factor=1.0):
    # get ax from figure
    ax = plt.gca()
    
    # set covariance matrix
    n_clusters = len(means) #High Time Resolution Universe Survey 
    # if covariance type is spherical, each component has its own single variance
    if cov_type == 'spherical':
        covs = [np.diag((covs[i],covs[i])) for i in range(n_clusters)]
    # if covariance type is diagonal, each component has its own diagonal covariance matrix
    elif cov_type == 'diag':
        covs = [np.diag((covs[i])) for i in range(n_clusters)]
    # if covariance type is tied, all components share the same general covariance matrix
    elif cov_type == 'tied':
        covs = [covs for _ in range(n_clusters)]
    # if covariance type is tied, each component has its own general covariance matrix
    elif cov_type == 'full':
        pass # nothing to do
    
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
                                  alpha=0.85)
        ell.set_clip_box(ax.bbox)
        ax.add_artist(ell)

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
 


#%% load toy dataset and visualize
title =  "clusters3"
path = "../datasets/toy/{}.csv".format(title)

data, true_labels = read_dataset(path, "label")
# display the scatter plot for this data set (without labels)
plt.figure()
plot_dataset(data, "f0", "f1", labels=None, title=title+" (unsupervised)")


#%% create a clustering model: GMM

# model hyper-parameters (values are up to you)
n_clusters = 5 
cov_type = "full"
 
# create the model 
gmm = GaussianMixture(n_components=n_clusters, 
                       covariance_type=cov_type,
                       random_state=22)

# train (fit) the model with data but without labels !!!
gmm.fit(data)

cluster_means = gmm.means_
cluster_covs = gmm.covariances_
labels = gmm.predict(data)

# reset plot interface
plt.figure()

# display data set 
plot_dataset(data, "f0", "f1", labels=labels, fill=False, 
             title="Number of Clusters: {}".format(n_clusters))
# display cluster centers
disp_cluster_means(cluster_means)
disp_cluster_ellipses(cluster_means, cluster_covs, cov_type)
# plot boundary
plot_boundary(gmm)
 

# model performance
sil_avg = silhouette_score(data, labels)
print("  > AVG Silhouette Score: {}".format(sil_avg))
lscore = gmm.score(data, labels)
print("  > AVG Likelihood Score: {}".format(lscore))
 


#%% create a clustering model: BGMM

# model hyper-parameters (values are up to you)
n_clusters = 5 
cov_type = "full"
btype = "dirichlet_distribution"
# gamma = 1e-9
gamma = 1e+3
# gamma = [1e-9, 1e3, 1e3, 1e3, 1e-9]
 
# create the model 
bgmm = BayesianGaussianMixture(n_components=n_clusters, 
                               covariance_type=cov_type,
                               weight_concentration_prior_type=btype,
                               weight_concentration_prior=gamma,
                               random_state=22)

# train (fit) the model with data but without labels !!!
bgmm.fit(data)

cluster_means = bgmm.means_
cluster_covs = bgmm.covariances_
labels = bgmm.predict(data)

# reset plot interface
plt.figure()

# display data set 
plot_dataset(data, "f0", "f1", labels=labels, fill=False, 
             title="Number of Clusters: {}".format(n_clusters))
# display cluster centers
disp_cluster_means(cluster_means)
disp_cluster_ellipses(cluster_means, cluster_covs, cov_type)
# plot boundary
plot_boundary(bgmm)
 

# model performance
sil_avg = silhouette_score(data, labels)
print("  > AVG Silhouette Score: {}".format(sil_avg))
lscore = bgmm.score(data, labels)
print("  > AVG Likelihood Score: {}".format(lscore))
 
