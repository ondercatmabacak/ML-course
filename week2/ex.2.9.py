#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:47:15 2019

@author: ayf

Example:
-------
The 1st Classification Algorithm: K-Nearest Neighbor

Model evaluation on TRAIN-TEST sets (real datasets)
"""

# import modules
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


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
    path = "../datasets/{}/{}.csv".format(dataset_name, dataset_name)
    data, labels = read_dataset(path, label_column)
    
    # describe complete data set
    print(">>> Complete Data Set: ")
    describe_dataset(data, labels)
    
    # split data set
    ratio = 0.3
    random_state = 22
    trainData, testData, trainLabels, testLabels = train_test_split(data, labels, 
                                                                    test_size=ratio, 
                                                                    random_state=random_state,
                                                                    shuffle=True)
    # describe TRAIN data set
    print(">>> Train Data: ")
    describe_dataset(trainData, trainLabels)
        
    # describe TEST data set
    print(">>> Test Data: ")
    describe_dataset(testData, testLabels)
    
    
    # fit model to TRAIN set
    knn.fit(trainData, trainLabels)
    
    #### model performance on TRAIN set
    # predict labels of train set
    predicted_train_labels = knn.predict(trainData)
    print(">>> Model Evaluation on TRAIN set")
    # evaluate model
    evaluate_model(knn, trainLabels, predicted_train_labels)
    
    
    #### model performance on TEST set
    # predict labels of test set
    predicted_test_labels = knn.predict(testData)
    print(">>> Model Evaluation on TEST set")
    # evaluate model
    evaluate_model(knn, testLabels, predicted_test_labels)

#%% create a classification model: KNN

# model hyper-parameters (values are up to you)
n_neigbors = 5 # how many neighbors will be checked during the prediction
minkowski_param = 2 # if p=1 then distance metric will be Manhattan, if p=2 than distance metric will be Euclidean

# create the model using the hyper parameters defined above
knn = KNeighborsClassifier(n_neighbors=n_neigbors, 
                           metric='minkowski', p=minkowski_param, 
                           n_jobs=3)


##################################################
################ REAL DATASETS  ##################
##################################################
#%% iris dataset
dataset_name =  "iris"
print("::::: Iris Dataset :::::\n")
test_model(knn, dataset_name, "species")

#%% digits dataset
dataset_name = "optdigits"
print("::::: Digits Dataset :::::\n")
test_model(knn, dataset_name, "digit")
 
#%% breast cancer dataset (Wisconsin Diagnostic Breast Cancer)
dataset_name = "wdbc"
print("::::: Breast Dataset :::::\n")
test_model(knn, dataset_name, "diagnosis")

#%% htru2 dataset (High Time Resolution Universe Collaboration using the Parkes Observatory)
dataset_name = "htru2"
print("::::: HTRU2 Dataset :::::\n")
test_model(knn, dataset_name, "ispulsar")

