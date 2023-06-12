#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:40:42 2019

@author: ayf

The 3rd Classification algorithm: Naive Bayes Classifier

Training model and predicting some novel instances
Displaying decision boundary
"""

# import modules
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB

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
   plt.scatter(point[:,0], point[:,1], marker="o", color="black", s=60, alpha=0.7)
   plt.text(point[0,0], point[0,1], "?", fontsize=15, 
            verticalalignment="bottom",
            horizontalalignment="left")

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
    
#%% create a classification model: Gaussian Naive Bayes
gnb = GaussianNB()

#%% load blobs dataset and visualize
title =  "binary_blobs"
path = "../datasets/toy/{}.csv".format(title)

data, labels = read_dataset(path, "label")
# display the scatter plot for this data set
plot_dataset(data, "f0", "f1", labels, title) 

#%% train (fit) the model with dataset
print(">>> Model is learning the data...")
gnb.fit(data, labels)
print(">>> Finished.\n")

#%% predict the label of a single data point

# determine a single point for testing
point = np.array([[-0.3, 1.7]])
# display the point 
disp_point(point)
# what should the label of that point be ? predict yourself.
# calculate the posterioor probability of the point with respect to each class
probs = gnb.predict_proba(point)
print ("")
print(">>> gnb says the probability of {} belonging to class-0 is {:.2f}"
      .format(point, probs[0,0]))
print(">>> gnb says the probability of {} belonging to class-1 is {:.2f}"
      .format(point, probs[0,1]))
# what gnb predicts ? of course the most likely one
predicted_label = gnb.predict(point)
print(">>> gnb says the label of {} is {}".format(point, predicted_label))

 
#%% predict the label of single data point

# determine a single point for testing
point = np.array([[-6, 4]])
# display the point 
disp_point(point)
# what should the label of that point be ? predict yourself.
# calculate the posterioor probability of the point with respect to each class
probs = gnb.predict_proba(point)
print ("")
print(">>> gnb says the probability of {} belonging to class-0 is {:.2f}"
      .format(point, probs[0,0]))
print(">>> gnb says the probability of {} belonging to class-1 is {:.2f}"
      .format(point, probs[0,1]))
# what gnb predicts ? of course the most likely one
predicted_label = gnb.predict(np.array(point))
print(">>> gnb says the label of {} is {}".format(point, predicted_label))


#%% plot boundary
plot_boundary(gnb)

 