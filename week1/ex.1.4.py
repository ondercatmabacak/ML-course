#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 01:43:12 2019

@author: ayf

Example:
--------
Read a dataset with pandas and examine it

We have many datasets for this course; toy datasets and real datasets
In this example, we will examine each real dataset
Real Datasets;
- iris dataset
- digits dataset
- breast cancer dataset
- htru2 dataset
"""


# import pyplot module as plt (module names are commonly shortened for simplicity) 
import matplotlib.pyplot as plt
# import numpy module as np (module names are commonly shortened for simplicity) 
import numpy as np
# import pandas modlule as pd (module names are commonly shortened for simplicity) 
import pandas as pd
# import scatter matrix from pandas
from pandas.plotting import scatter_matrix


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
        label_text = "class: {}".format(label)
        # scatter datapoints belonging to the 'label'
        plt.scatter(x_points, y_points, marker="o", alpha=0.70, label=label_text)
    
    # place legend box at the best location
    plt.legend(loc="best", fontsize=14)
    # display labels for x and y axes
    plt.xlabel(col1, size=14)
    plt.ylabel(col2, size=14)


##################################################
################ REAL DATASETS 
##################################################
#%% iris dataset
path = "../datasets/iris/iris.csv"

print("::::: Iris Dataset :::::\n")
data, labels = read_dataset(path, "species")

'''
150 data instances, 4 features (sepal_length, sepal_width, petal_length, petal_width)
3 classes (labels); setosa, versicolor, virginica (species of iris plant)
50 instances for class setosa
50 instances for class versicolor
50 instances for class virginica
'''
# using pands, we can display histogram of all features
# plt.figure()
_ = data.hist()
_ = plt.suptitle("Histogram of Sepal Length")

# using 'scatter_matrix' method, you can see the scatter plot of each pair of features
# if there is a correlation between any pair of features, it can be seen on scatter plot 
# off-diagonal plots are scatter plots 
# diagonal plots can be set as histograms or estimated density
#plt.figure()
_ = scatter_matrix(data, diagonal="kde")
_ = plt.suptitle("Diagonals are histogram")

#%% optical digits dataset 
path = "../datasets/optdigits/optdigits.csv"

print("::::: Optical Digits Dataset  :::::\n")
data, labels = read_dataset(path, "digit")

'''
5620 data instances, 64 features (f00, f01, ..., f77)
10 classes (labels); 0, 1, 2, ..., 9 (digits)
554 instances for class 0
571 instances for class 1
557 instances for class 2
572 instances for class 3
568 instances for class 4
558 instances for class 5
558 instances for class 6
566 instances for class 7
554 instances for class 8
562 instances for class 9
'''
 
#%% breast cancer dataset (Wisconsin Diagnostic Breast Cancer)
path = "../datasets/wdbc/wdbc.csv"

print("::::: Breast Dataset (Test) :::::\n")
data, labels = read_dataset(path, "diagnosis")

'''
569 data instances, 30 features  
2 classes (labels); M (Malignant) and B (Benign)
357 instances for class B
212 instances for class M
''' 

# display any two feature
plot_dataset(data, "radius error", "texture error", labels, "breast_cancer")
# display any two feature
plot_dataset(data, "mean symmetry", "worst compactness", labels, "breast_cancer")

#%% htru2 dataset (High Time Resolution Universe Collaboration using the Parkes Observatory)
path = "../datasets/htru2/htru2.csv"

print("::::: HTRU2 Dataset (Test) :::::\n")
data, labels = read_dataset(path, "ispulsar")

'''
17898 data instances, 8 features ('pmean', 'pstd', 'pskew', 'pkurt', 'dmean', 'dstd', 'dskew', 'dkur') 
2 classes (labels); 0 (not pulsar) and 1 (pulsar)
16259 instances for class 0
1639 instances for class 1
''' 

# display any two feature
plot_dataset(data, "pmean", "dmean", labels, "htru2")
# display any two feature
plot_dataset(data, "pstd", "dstd", labels, "htru2")