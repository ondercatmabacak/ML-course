#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:50:43 2019

@author: ayf

Example:
-------
The 3rd Classification algorithm: Naive Bayes Classifier

Model performance on toy datasets
"""
 
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from scipy import linalg

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
    h = .01  # step size in the mesh grid (if h is too low, then there might be memory error !)
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
        

# plot Gaussian ellipses
def plot_ellipses(model, factor=2.0):
    # estimated Gaussian params ?
    means  = model.theta_ # per class
    sigmas = model.sigma_ # per class
    
    
    # for each class, draw Gaussian ellipse with 'factor' confidence level
    for mean, sigma in zip(means, sigmas):
        
        # first, display the mean point
        disp_point(mean)
        
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
                                  linestyle="--", linewidth=2, 
                                  edgecolor="black", facecolor='none', alpha=0.5, zorder=-30)
        ax = plt.gca()
        ell.set_clip_box(ax.bbox)
        ax.add_artist(ell)


closeAll()

#%% create a classification model: Gaussian Naive Bayes (Gaussian NB)
gnb = GaussianNB()

##################################################
################ TOY DATASETS  ###################
##################################################
#%% binary labeled blobs dataset
title =  "binary_blobs"
path = "../datasets/toy/{}.csv".format(title)

print("::::: Binary Labeled Blobs Dataset :::::\n")
data, labels = read_dataset(path, "label")
plot_dataset(data, "f0", "f1", labels, title)

# train model
gnb.fit(data, labels)
# make predictions
preds = gnb.predict(data)
# plot boundary
plot_boundary(gnb)
# plot gaussian ellipses
plot_ellipses(gnb)

acc_score = accuracy_score(labels, preds)
print(">>> Accuracy Score: {}".format(acc_score))
conf_matrix = confusion_matrix(labels, preds)
print(">>> Confusion matrix: \n{}".format(conf_matrix))
report = classification_report(labels, preds)
print(">>> Classification Report: \n{}".format(report))
print("")
 

#%% binary labeled circles dataset
title =  "binary_circles"
path = "../datasets/toy/{}.csv".format(title)

print("::::: Binary Labeled Circles Dataset :::::\n")
data, labels = read_dataset(path, "label")
plot_dataset(data, "f0", "f1", labels, title)

# train model
gnb.fit(data, labels)
# make predictions
preds = gnb.predict(data)
# plot boundary
plot_boundary(gnb)
# plot gaussian ellipses
plot_ellipses(gnb)

acc_score = accuracy_score(labels, preds)
print(">>> Accuracy Score: {}".format(acc_score))
conf_matrix = confusion_matrix(labels, preds)
print(">>> Confusion matrix: \n{}".format(conf_matrix))
report = classification_report(labels, preds)
print(">>> Classification Report: \n{}".format(report))
print("")
 

#%% binary labeled moons dataset
title =  "binary_moons"
path = "../datasets/toy/{}.csv".format(title)

print("::::: Binary Labeled Moons Dataset :::::\n")
data, labels = read_dataset(path, "label")
plot_dataset(data, "f0", "f1", labels, title)

# train model
gnb.fit(data, labels)
# make predictions
preds = gnb.predict(data)
# plot boundary
plot_boundary(gnb)
# plot gaussian ellipses
plot_ellipses(gnb)

acc_score = accuracy_score(labels, preds)
print(">>> Accuracy Score: {}".format(acc_score))
conf_matrix = confusion_matrix(labels, preds)
print(">>> Confusion matrix: \n{}".format(conf_matrix))
report = classification_report(labels, preds)
print(">>> Classification Report: \n{}".format(report))
print("")
 

#%% binary labeled intermixed dataset
title =  "binary_intermixed"
path = "../datasets/toy/{}.csv".format(title)

print("::::: Binary Labeled Intermixed Dataset :::::\n")
data, labels = read_dataset(path, "label")
plot_dataset(data, "f0", "f1", labels, title)

# train model
gnb.fit(data, labels)
# make predictions
preds = gnb.predict(data)
# plot boundary
plot_boundary(gnb)
# plot gaussian ellipses
plot_ellipses(gnb)

acc_score = accuracy_score(labels, preds)
print(">>> Accuracy Score: {}".format(acc_score))
conf_matrix = confusion_matrix(labels, preds)
print(">>> Confusion matrix: \n{}".format(conf_matrix))
report = classification_report(labels, preds)
print(">>> Classification Report: \n{}".format(report))
print("")
 

#%% multi labeled blobs dataset
title =  "multilabel_blobs"
path = "../datasets/toy/{}.csv".format(title)

print("::::: Multi Labeled Blobs Dataset :::::\n")
data, labels = read_dataset(path, "label")
plot_dataset(data, "f0", "f1", labels, title)

# train model
gnb.fit(data, labels)
# make predictions
preds = gnb.predict(data)
# plot boundary
plot_boundary(gnb)
# plot gaussian ellipses
plot_ellipses(gnb)

acc_score = accuracy_score(labels, preds)
print(">>> Accuracy Score: {}".format(acc_score))
conf_matrix = confusion_matrix(labels, preds)
print(">>> Confusion matrix: \n{}".format(conf_matrix))
report = classification_report(labels, preds)
print(">>> Classification Report: \n{}".format(report))
print("")
 

#%% multi labeled intermixed dataset
title =  "multilabel_intermixed"
path = "../datasets/toy/{}.csv".format(title)

print("::::: Multi Labeled Intermixed Dataset :::::\n")
data, labels = read_dataset(path, "label")
plot_dataset(data, "f0", "f1", labels, title)

# train model
gnb.fit(data, labels)
# make predictions
preds = gnb.predict(data)
# plot boundary
plot_boundary(gnb)
# plot gaussian ellipses
plot_ellipses(gnb)

acc_score = accuracy_score(labels, preds)
print(">>> Accuracy Score: {}".format(acc_score))
conf_matrix = confusion_matrix(labels, preds)
print(">>> Confusion matrix: \n{}".format(conf_matrix))
report = classification_report(labels, preds)
print(">>> Classification Report: \n{}".format(report))
print("")
 


