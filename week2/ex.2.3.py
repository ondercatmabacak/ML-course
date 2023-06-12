#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:17:43 2019

@author: ayf

Example:
-------
The 1st Classification Algorithm: K-Nearest Neighbor

Examinining model performance
"""

# import modules
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


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
    # plot grid points with their predicted labels (by KNN) that corresponds to a color
    plt.imshow(Z, interpolation="gaussian", zorder=-100, 
               alpha=0.3, extent=[x_min, x_max, y_min, y_max], 
               aspect="auto", origin="lower")
   
#%% create a classification model: KNN

# model hyper-parameters (values are up to you)
n_neigbors = 10 # how many neighbors will be checked during the prediction
minkowski_param = 1 # if p=1 then distance metric will be Manhattan, if p=2 than distance metric will be Euclidean

# create the model using the hyper parameters defined above
knn = KNeighborsClassifier(n_neighbors=n_neigbors, 
                           metric='minkowski', p=minkowski_param, 
                           n_jobs=3)
# n_jobs -> how many threads for parallel computing ? (it can be equal to # of cores - 1)

#%% KNN on binary labeled intermixed dataset
title =  "binary_intermixed"
path = "../datasets/toy/{}.csv".format(title)

print("::::: Binary Labeled Intermixed Dataset :::::\n")
data, labels = read_dataset(path, "label")
# display the scatter plot for this data set
plot_dataset(data, "f0", "f1", labels, title) 

# 1st step: train (fit) the model with dataset
print(">>> Model is learning the {}".format(title))
knn.fit(data, labels)
print(">>> Finished.")

# display boundaries
plot_boundary(knn)

# 2nd step: predict all labels of the train set (dataset)
preds = knn.predict(data)

# what did we do ?
# 1st step: we let KNN to learn the data
# 2nd step: we let KNN to predict the label of each instances in data that KNN has learned before
# Did KNN learn the dataset perfectly ? 
# compare the actual labels and the predicted labels to evaluate the model performance on training set (data set)

# how many data point labels are predicted correctly, and how many of them are predicted wrongly ?
number_of_true_predictions = 0
number_of_false_predictions = 0
false_predicted_points = []

for idx, (real_label, predicted_label) in enumerate(zip(labels, preds)):
    # if prediction is true, then no problem
    if real_label == predicted_label:
        # increase the counter
        number_of_true_predictions += 1
    # if prediction is false, then add the point to the list (false_predicted_points)
    else:
        number_of_false_predictions += 1
        point = data.loc[idx]
        false_predicted_points.append(point)
# convert data type 
false_predicted_points = np.array(false_predicted_points)

print(">>> Model performance:")
print("  > Number of true  predictions: {}".format(number_of_true_predictions))
print("  > Number of false predictions: {}".format(number_of_false_predictions))
 
# As you see, KNN is not 'perfecly' fitted to the dataset (15 false predictions)
# Which data points are falsely predicted ?
plt.scatter(false_predicted_points[:,0], false_predicted_points[:,1], 
            marker="o", edgecolors="black", facecolors='none', 
            s=120, label="Falsely Predicted")
plt.legend()


#%% KNN on intermixed dataset
title =  "multilabel_intermixed"
path = "../datasets/toy/{}.csv".format(title)

print("::::: Multi Labeled Intermixed Dataset :::::\n")
data, labels = read_dataset(path, "label")
# display the scatter plot for this data set
plot_dataset(data, "f0", "f1", labels, title) 

# 1st step: train (fit) the model with dataset
print(">>> Model is learning the {}".format(title))
knn.fit(data, labels)
print(">>> Finished.")
# display boundaries
plot_boundary(knn)
# 2nd step: predict all labels of the train set (dataset)
preds = knn.predict(data)
# how many data point labels are predicted correctly, and how many of them are predicted wrongly ?
number_of_true_predictions = 0
number_of_false_predictions = 0
false_predicted_points = []

for idx, (real_label, predicted_label) in enumerate(zip(labels, preds)):
    # if prediction is true, then no problem
    if real_label == predicted_label:
        # increase the counter
        number_of_true_predictions += 1
    # if prediction is false, then add the point to the list (false_predicted_points)
    else:
        number_of_false_predictions += 1
        point = data.loc[idx]
        false_predicted_points.append(point)
# convert data type 
false_predicted_points = np.array(false_predicted_points)

print(">>> Model performance:")
print("  > Number of true  predictions: {}".format(number_of_true_predictions))
print("  > Number of false predictions: {}".format(number_of_false_predictions))
 
# As you see, KNN is not 'perfecly' fitted to the dataset (15 false predictions)
# Which data points are falsely predicted ?
plt.scatter(false_predicted_points[:,0], false_predicted_points[:,1], 
            marker="o", edgecolors="black", facecolors='none', 
            s=120, label="Falsely Predicted")
plt.legend()

# so is there a metric for evaluating model performance ?