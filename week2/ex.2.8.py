#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 07:18:19 2019

@author: ayf

Example:
-------
The 1st Classification Algorithm: K-Nearest Neighbor

Model evaluation on TRAIN-TEST sets (toy datasets)
"""

# import modules
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# a method for closing all plot windows 
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
def plot_dataset(data, col1, col2, labels, marker, title=""):
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
        label_text = "class: {}".format(label)
        # scatter datapoints belonging to the 'label'
        plt.scatter(x_points, y_points, marker=marker, alpha=0.70, label=label_text)
    
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

# method for describing data set
def describe_dataset(data, labels):
    # print data shape 
    print("  > Data shape: {}".format(data.shape))
    # how many labels for each class
    for label in np.unique(labels):
        count = labels[labels==label].shape[0]
        print("  > {} instances for class {}".format(count, label))
    print("")

# method for evaluating model
def evaluate_model(model, true_labels, predicted_labels):
    acc_score = accuracy_score(true_labels, predicted_labels)
    print("  > Accuracy Score: {}".format(acc_score))
    report = classification_report(true_labels, predicted_labels)
    print("  > Classification Report: \n{}".format(report))
    print("")


# method for testing a model on a dataset
def test_model(model, dataset_name, label_column):
    '''
        'model' is a created KNN model
        'dataset_name' is the name of the dataset to be examined
        'label_column' is the column name of labels in data set 
    '''
    # read dataset
    path = "../datasets/{}/{}.csv".format("toy", dataset_name)
    data, labels = read_dataset(path, label_column)
    
    # describe & plot complete data set
    print(">>> Complete Data Set: ")
    describe_dataset(data, labels)
    plot_dataset(data, "f0", "f1", labels, "o", dataset_name)
    # get axes limits
    xlim, ylim = plt.xlim(), plt.ylim()
    
    # split data set
    ratio = 0.3
    random_state = 22
    trainData, testData, trainLabels, testLabels = train_test_split(data, labels, 
                                                                    test_size=ratio, 
                                                                    random_state=random_state,
                                                                    shuffle=True)
    # describe & plot TRAIN data set
    print(">>> Train Data: ")
    describe_dataset(trainData, trainLabels)
    title1 = dataset_name+" (train)"
    plot_dataset(trainData, "f0", "f1", trainLabels, "o", title1)
    _ = plt.xlim(xlim), plt.ylim(ylim)
    
    # describe & plot TEST data set
    print(">>> Test Data: ")
    describe_dataset(testData, testLabels)
    title2 = dataset_name+" (test)"
    plot_dataset(testData, "f0", "f1", testLabels, "x", title2)
    _ = plt.xlim(xlim), plt.ylim(ylim)
    
    # fit model to TRAIN set
    knn.fit(trainData, trainLabels)
    
    #### model performance on TRAIN set
    # predict labels of train set
    predicted_train_labels = knn.predict(trainData)
    print(">>> Model Evaluation on TRAIN set")
    evaluate_model(knn, trainLabels, predicted_train_labels)
    # display decision boundary
    plot_boundary(knn) 
    # plot train set
    plot_dataset(trainData, "f0", "f1", trainLabels, "o", title1)
    _ = plt.xlim(xlim), plt.ylim(ylim)
    
    #### model performance on TEST set
    # predict labels of test set
    predicted_test_labels = knn.predict(testData)
    print(">>> Model Evaluation on TEST set")
    evaluate_model(knn, testLabels, predicted_test_labels)
    # display decision boundary
    plot_boundary(knn) 
    # plot test set
    plot_dataset(testData, "f0", "f1", testLabels, "x", title2)
    _ = plt.xlim(xlim), plt.ylim(ylim)
 


#%% create a classification model: KNN

# model hyper-parameters (values are up to you)
n_neigbors = 5 # how many neighbors will be checked during the prediction
minkowski_param = 2 # if p=1 then distance metric will be Manhattan, if p=2 than distance metric will be Euclidean

# create the model using the hyper parameters defined above
knn = KNeighborsClassifier(n_neighbors=n_neigbors, 
                           metric='minkowski', p=minkowski_param, 
                           n_jobs=3)


##################################################
################ TOY DATASETS  ###################
##################################################
#%% binary labeled blobs dataset
closeAll()
dataset_name =  "binary_blobs"
print("::::: Binary Labeled Blobs Dataset :::::\n")
test_model(knn, dataset_name, "label")
 
#%% binary labeled circles dataset
closeAll()
dataset_name =  "binary_circles"
print("::::: Binary Labeled Circles Dataset :::::\n")
test_model(knn, dataset_name, "label")

#%% binary labeled moons dataset
closeAll()
dataset_name =  "binary_moons"
print("::::: Binary Labeled Moons Dataset :::::\n")
test_model(knn, dataset_name, "label")

#%% binary labeled intermixed dataset
closeAll()
dataset_name =  "binary_intermixed"
print("::::: Binary Labeled Intermixed Dataset :::::\n")
test_model(knn, dataset_name, "label")

#%% multi labeled blobs dataset
closeAll()
dataset_name =  "multilabel_blobs"
print("::::: Multi Labeled Blobs Dataset :::::\n")
test_model(knn, dataset_name, "label")

#%% multi labeled intermixed dataset
closeAll()
dataset_name =  "multilabel_intermixed"
print("::::: Multi Labeled Intermixed Dataset :::::\n")
test_model(knn, dataset_name, "label")

 