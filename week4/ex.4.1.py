#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:26:52 2019

@author: ayf

The 3rd Classification algorithm: Naive Bayes Classifier

Examining estimated parameters (prior, mean, sigma)
"""

# import modules
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from scipy import linalg
import matplotlib as mpl
from matplotlib import pyplot as plt


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
   plt.scatter([point[0]], [point[1]], marker="*", color="black", s=90, alpha=0.7)

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
               alpha=0.3, extent=[x_min, x_max, y_min, y_max], 
               aspect="auto", origin="lower") 
        
closeAll()

#%% create a classification model: Gaussian Naive Bayes (Gaussian NB)
gnb = GaussianNB()

#%% load blobs dataset and visualize
title =  "binary_blobs"
path = "../datasets/toy/{}.csv".format(title)

data, labels = read_dataset(path, "label")
# display the scatter plot for this data set
plot_dataset(data, "f0", "f1", labels, title) 

# get unique labels
unique_labels = labels.unique()

#%% train (fit) the model with dataset
print(">>> Model is learning the data...")
gnb.fit(data, labels)
print(">>> Finished.")

#%% what parameters are learned from the data ?

print(">>> How many instances per class ?")
for label, count in zip(unique_labels, gnb.class_count_):
    print ("  > {} instances for class-{}".format(count, label))
print("") 

print(">>> Prior probability of per class ?")
for label, prior_prob in zip(unique_labels, gnb.class_prior_):
    print ("  > class-{} prior probability: {}".format(label, prior_prob))
print("") 

print(">>> Mean of per class ?")
for label, mean in zip(unique_labels, gnb.theta_):
    print ("  > class-{} mean: {}".format(count, mean))
    disp_point(mean)
print("") 

print(">>> Variance of per class ?")
for label, variance in zip(unique_labels, gnb.sigma_):
    print ("  > class-{} variance: {}".format(count, variance))
print("") 


#%% plot boundary
plot_boundary(gnb)
    

#%% display Gaussian ellipses for each class
# confidence level ?
factor = 2.0

# estimated Gaussian params ?
means  = gnb.theta_ # per class
sigmas = gnb.sigma_ # per class

# for each class, draw Gaussian ellipse with 'factor' confidence level
for mean, sigma in zip(means, sigmas):
    # convert variance vector to variance-covariance matrix
    # note that covariance is ZERO (because the features are assumed to be independent)
    covar = [[sigma[0], 0.0],
             [0.0      , sigma[1]]]
    
    # calculate eigenvectors and eigenvalues
    eig_vec,eig_val,_ = np.linalg.svd(covar)
    # Make sure 0th eigenvector has positive x-coordinate
    if eig_vec[0][0] < 0:
        eig_vec[0] *= -1
    # calculate major and minor axes length
    majorLength = 2*factor*np.sqrt(eig_val[0])
    minorLength = 2*factor*np.sqrt(eig_val[1])
    # calculate rotation angle
    u = eig_vec[0] / linalg.norm(eig_vec[0])
    angle = np.arctan(u[1]/u[0])
    angle = 180.0*angle/np.pi
    
    # create ellipse
    ell = mpl.patches.Ellipse(mean, majorLength, minorLength, angle, 
                              linestyle="-", linewidth=2, 
                              edgecolor="black", facecolor='none', alpha=0.7, zorder=30)
    ax = plt.gca()
    ell.set_clip_box(ax.bbox)
    ax.add_artist(ell)

