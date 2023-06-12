#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:49:45 2019

@author: ayf

How to determine covariance type ?
silhouette analysis
likelihood score analysis
AIC analysis
BIC analysis
"""

# import modules
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
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
def disp_cluster_ellipses(means, covs, cov_type, factor=2.0):
    # get ax from figure
    ax = plt.gca()
    
    # set covariance matrix
    n_clusters = len(means)
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
                                  alpha=0.85, zorder=-100)
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
    plt.imshow(Z, interpolation="gaussian",  
               alpha=0.2, extent=[x_min, x_max, y_min, y_max], 
               aspect="auto", origin="lower")
    
# method for asking user input (for blocking current operation)
def pause():
    plt.draw()
    plt.pause(0.20)
    raw_input("Press enter to continue ...")


#%% load toy dataset and visualize
title =  "clusters2"
path = "../datasets/toy/{}.csv".format(title)

data, true_labels = read_dataset(path, "label")
# display the scatter plot for this data set (without labels)
plt.figure()
plot_dataset(data, "f0", "f1", labels=None, title=title+" (unsupervised)")


#%% create a clustering model: GMM

# model hyper-parameters (values are up to you)
n_clusters = 7 
covs = ["spherical", "tied", "diag", "full"]

# dict variable to keep created models and scores 
result = {}
#################
for cov_type in covs:
    # pause the code
    # pause()
    
    print("====================")
    print(">>> Covariance Type: {}".format(cov_type))
    
    # create the model 
    gmm = GaussianMixture(n_components=n_clusters, covariance_type=cov_type,
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
                 title="Covariance Type: {}".format(cov_type))
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
    aic_score = gmm.aic(data)
    print("  > AIC Score: {}".format(aic_score))
    bic_score = gmm.bic(data)
    print("  > BIC Score: {}".format(bic_score))
    
    # keep the scores
    result[cov_type] = {"model": gmm,
                          "scores": {"sil_score": sil_avg,
                                      "lscore": lscore,
                                      "aic_score":aic_score,
                                      "bic_score":bic_score
                                    }
                          }
    

#%% print scores with respect to "number of clusters"

print("======================================")
for cov_type in covs:
    print(">>> cov_type: {}".format(cov_type))
    for score_type in result[cov_type]["scores"].keys():
        score = result[cov_type]["scores"][score_type]
        print ("    {}: {: .2f}".format(score_type, score))

print("======================================")

#%% Silhouette Score Analysis

# get silhouette score list
sil_list = [result[n]["scores"]["sil_score"] for n in covs]

# create figure
plt.figure()
# get axis
ax1 = plt.gca()
# plot the silhouette score 
ax1.scatter(covs, sil_list, color="tab:blue", label="silhouette score")
ax1.plot(covs, sil_list, color="tab:blue")

# edit labels and other staffs
ax1.set_xlabel("covariance type", size=14)
ax1.set_ylabel("silhouette score", size=14)
ax1.legend(loc="best")
ax1.set_xticks(covs)
ax1.legend(loc='upper left')
ax1.grid(axis="x")

#%% AVG Log Likelihood Score Analysis

# get average log likelihood score list
lscore_list = [result[n]["scores"]["lscore"] for n in covs]

# create figure
plt.figure()
# get axis
ax1 = plt.gca()
# plot the average log likelihood score 
ax1.scatter(covs, lscore_list, color="tab:blue", label="avg log likelihood score")
ax1.plot(covs, lscore_list, color="tab:blue")

# edit labels and other staffs
ax1.set_xlabel("covariance type", size=14)
ax1.set_ylabel("avg log likelihood score", size=14)
ax1.legend(loc="best")
ax1.set_xticks(covs)
ax1.legend(loc='upper left')
ax1.grid(axis="x")

#%% AIC analysis

# get AIC score list
aic_score_list = [result[n]["scores"]["aic_score"] for n in covs]

# create figure
plt.figure()
# get axis
ax1 = plt.gca()
# plot AIC  score 
ax1.scatter(covs, aic_score_list, color="tab:blue", label="AIC score")
ax1.plot(covs, aic_score_list, color="tab:blue")

# edit labels and other staffs
ax1.set_xlabel("n_clusters", size=14)
ax1.set_ylabel("avg log likelihood score", size=14)
ax1.legend(loc="best")
ax1.set_xticks(covs)
ax1.legend(loc='upper left')
ax1.grid(axis="x")


#%% BIC analysis
 
# get BIC score list
bic_score_list = [result[n]["scores"]["bic_score"] for n in covs]

# create figure
plt.figure()
# get axis
ax1 = plt.gca()
# plot BIC scores 
ax1.scatter(covs, bic_score_list, color="tab:blue", label="BIC score")
ax1.plot(covs, bic_score_list, color="tab:blue")

# edit labels and other staffs
ax1.set_xlabel("n_clusters", size=14)
ax1.set_ylabel("BIC score", size=14)
ax1.legend(loc="best")
ax1.set_xticks(covs)
ax1.legend(loc='upper left')
ax1.grid(axis="x")


#%% determining best model

# highest silhouette score is better
# highest AVG log likelihood is better
# lowest AIC or BIC score is better
 
best_cov_type = "tied" 
# get model from results according to the best number of clusters
gmm = result[best_cov_type]["model"]
# get scores
scores = result[best_cov_type]["scores"]

# get estimated params
cluster_means = gmm.means_
cluster_covs = gmm.covariances_
# get predictions
labels = gmm.predict(data)

# reset plot interface
plt.figure()

# display data set 
plot_dataset(data, "f0", "f1", labels=labels, fill=False, 
             title="Covariance Type: {}".format(best_cov_type))
# display cluster centers
disp_cluster_means(cluster_means)
disp_cluster_ellipses(cluster_means, cluster_covs, best_cov_type)
# plot boundary
plot_boundary(gmm)

print(">>> Scores")
for score_type in scores.keys():
    score = scores[score_type]
    print ("    {}: {: .2f}".format(score_type, score))
#%% display the data with true labels
# display the scatter plot for this data set (with labels)
plt.figure()
plot_dataset(data, "f0", "f1", true_labels, title=title+" (supervised)")
plot_boundary(gmm) 