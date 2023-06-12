#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:28:02 2019

@author: ayf

Example:
--------
Read a dataset with pandas and examine it

We have many datasets for this course; toy datasets and real datasets
In this example, we will examine each toy dataset
Toy Datasets;
- binary labeled blobs dataset
- binary labeled circles dataset
- binary labeled moons dataset
- binary labeled intermixed dataset
- multi labeled blobs dataset
- multi labeled intermixed dataset
"""

# import pyplot module as plt (module names are commonly shortened for simplicity) 
import matplotlib.pyplot as plt
# import numpy module as np (module names are commonly shortened for simplicity) 
import numpy as np
# import pandas modlule as pd (module names are commonly shortened for simplicity) 
import pandas as pd

# invoke this method on ipython console when there are lots of plots
def closeAll():
    plt.close("all")

# method for reading and describing datasets
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
    
    # print data shape (150 data points with 4 features)
    print(">>> Data shape: {}".format(data.shape))
    # how many features in the dataset
    print(">>> Features: {}".format(features))
    # how many labels in the dataset
    print(">>> Unique Labels: {}".format(np.unique(labels)))
    # how many labels for each class
    for label in np.unique(labels):
        count = labels[labels==label].shape[0]
        print("  > {} instances for class {}".format(count, label))
    print("")
    
    # get the first 5 rows of the data set with labels
    dataset_first5 = dataset.head(5)
    print(">>> First 5 elements of dataset: ")
    print(dataset_first5)
    print("")
    
    # get the last 5 elements of the data set with labels
    dataset_last5 = dataset.tail(5)
    print(">>> Last 5 elements of dataset: ")
    print(dataset_last5)
    print("")
    
    # sample 5 elements from dataset randomly 
    dataset_sample5 = dataset.sample(5)
    print(">>> 5 random elements of dataset: ")
    print(dataset_sample5)
    print("")
    
    # get statistical summary of the dataset 
    stats = dataset.describe() #to give a statistical summary about the dataset
    print(">>> Statistical summary of dataset: ")
    print(stats)
    print("")
    
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
    for label in labels.unique():
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



##################################################
################ TOY DATASETS 
##################################################
#%% binary labeled blobs dataset
title =  "binary_blobs"
path = "../datasets/toy/{}.csv".format(title)

print("::::: Binary Labeled Blobs Dataset :::::\n")
data, labels = read_dataset(path, "label")

'''
200 data instances, 2 features (f0 and f1)
2 classes (labels); 0 and 1
100 instances for class-0
100 instances for class-1
'''
# display the scatter plot for this data set
plot_dataset(data, "f0", "f1", labels, title)

#%% binary labeled circles dataset
title =  "binary_circles"
path = "../datasets/toy/{}.csv".format(title)

print("::::: Binary Labeled Circles Dataset :::::\n")
data, labels = read_dataset(path, "label")

'''
200 data instances, 2 features (f0 and f1)
2 classes (labels); 0 and 1
100 instances for class-0
100 instances for class-1
'''
# display the scatter plot for this data set
plot_dataset(data, "f0", "f1", labels, title)

#%% binary labeled moons dataset
title =  "binary_moons"
path = "../datasets/toy/{}.csv".format(title)

print("::::: Binary Labeled Moons Dataset :::::\n")
data, labels = read_dataset(path, "label")

'''
200 data instances, 2 features (f0 and f1)
2 classes (labels); 0 and 1
100 instances for class-0
100 instances for class-1
'''
# display the scatter plot for this data set
plot_dataset(data, "f0", "f1", labels, title)

#%% binary labeled intermixed dataset
title =  "binary_intermixed"
path = "../datasets/toy/{}.csv".format(title)

print("::::: Binary Labeled Intermixed Dataset :::::\n")
data, labels = read_dataset(path, "label")

'''
200 data instances, 2 features (f0 and f1)
2 classes (labels); 0 and 1
99 instances for class-0
101 instances for class-1
'''
# display the scatter plot for this data set
plot_dataset(data, "f0", "f1", labels, title)

#%% multi labeled blobs dataset
title =  "multilabel_blobs"
path = "../datasets/toy/{}.csv".format(title)

print("::::: Multi Labeled Blobs Dataset :::::\n")
data, labels = read_dataset(path, "label")

'''
300 data instances, 2 features (f0 and f1)
3 classes (labels); 0, 1 and 2
100 instances for class-0
100 instances for class-1
100 instances for class-2
'''
# display the scatter plot for this data set
plot_dataset(data, "f0", "f1", labels, title)

#%% multi labeled intermixed dataset
title =  "multilabel_intermixed"
path = "../datasets/toy/{}.csv".format(title)

print("::::: Multi Labeled Intermixed Dataset :::::\n")
data, labels = read_dataset(path, "label")

'''
300 data instances, 2 features (f0 and f1)
3 classes (labels); 0, 1 and 2
101 instances for class-0
101 instances for class-1
98 instances for class-2
'''
# display the scatter plot for this data set
plot_dataset(data, "f0", "f1", labels, title)

