#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:56:08 2019

@author: ayf

Example:
-------
The 3rd Classification algorithm: Naive Bayes Classifier

Model evaluation on TRAIN-TEST sets (toy datasets)
"""

# import modules
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib as mpl
from matplotlib import pyplot as plt
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
    plt.draw()

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
    # plot grid points with their predicted labels (by dtc) that corresponds to a color
    plt.imshow(Z, interpolation="gaussian", zorder=-100, 
               alpha=0.3, extent=[x_min, x_max, y_min, y_max], 
               aspect="auto", origin="lower")
    plt.draw()

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
        'model' is a created dtc model
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
    model.fit(trainData, trainLabels)
    
    #### model performance on TRAIN set
    # predict labels of train set
    predicted_train_labels = model.predict(trainData)
    print(">>> Model Evaluation on TRAIN set")
    evaluate_model(model, trainLabels, predicted_train_labels)
    # plot train set
    plot_dataset(trainData, "f0", "f1", trainLabels, "o", title1)
    _ = plt.xlim(xlim), plt.ylim(ylim)
    # display decision boundary
    plot_boundary(model) 
    # plot gaussian ellipses
    plot_ellipses(gnb)
    
    #### model performance on TEST set
    # predict labels of test set
    predicted_test_labels = model.predict(testData)
    print(">>> Model Evaluation on TEST set")
    evaluate_model(model, testLabels, predicted_test_labels)
    # plot test set
    plot_dataset(testData, "f0", "f1", testLabels, "x", title2)
    _ = plt.xlim(xlim), plt.ylim(ylim)
    # display decision boundary
    plot_boundary(model) 
    # plot gaussian ellipses
    plot_ellipses(gnb)
    
    
closeAll()

#%% create a classification model: Gaussian Naive Bayes (Gaussian NB)
gnb = GaussianNB()


##################################################
################ TOY DATASETS  ###################
##################################################
#%% binary labeled blobs dataset
closeAll()
dataset_name =  "binary_blobs"
print("::::: Binary Labeled Blobs Dataset :::::\n")
test_model(gnb, dataset_name, "label")
 
#%% binary labeled circles dataset
closeAll()
dataset_name =  "binary_circles"
print("::::: Binary Labeled Circles Dataset :::::\n")
test_model(gnb, dataset_name, "label")

#%% binary labeled moons dataset
closeAll()
dataset_name =  "binary_moons"
print("::::: Binary Labeled Moons Dataset :::::\n")
test_model(gnb, dataset_name, "label")

#%% binary labeled intermixed dataset
closeAll()
dataset_name =  "binary_intermixed"
print("::::: Binary Labeled Intermixed Dataset :::::\n")
test_model(gnb, dataset_name, "label")

#%% multi labeled blobs dataset
closeAll()
dataset_name =  "multilabel_blobs"
print("::::: Multi Labeled Blobs Dataset :::::\n")
test_model(gnb, dataset_name, "label")

#%% multi labeled intermixed dataset
closeAll()
dataset_name =  "multilabel_intermixed"
print("::::: Multi Labeled Intermixed Dataset :::::\n")
test_model(gnb, dataset_name, "label")
