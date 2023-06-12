#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:52:36 2019

@author: ayf

Example:
-------
The 1st Classification Algorithm: K-Nearest Neighbor

Evaluation metrics
"""

# import modules
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier


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


#%% load intermixed dataset
title =  "binary_intermixed"
path = "../datasets/toy/{}.csv".format(title)

print("::::: Binary Labeled Intermixed Dataset :::::\n")
data, labels = read_dataset(path, "label")


#%% create a classification model: KNN

# model hyper-parameters (values are up to you)
n_neigbors = 10 # how many neighbors will be checked during the prediction
minkowski_param = 1 # if p=1 then distance metric will be Manhattan, if p=2 than distance metric will be Euclidean

# create the model using the hyper parameters defined above
knn = KNeighborsClassifier(n_neighbors=n_neigbors, 
                           metric='minkowski', p=minkowski_param, 
                           n_jobs=3)

# train (fit) the model with dataset
print(">>> Model is learning the {}".format(title))
knn.fit(data, labels)
# now, make predictions on dataset
print(">>> Model is predicting labels of {}".format(title))
predicted_labels = knn.predict(data)

# at the previous example, we counted the number of true and false predictions
# there are some methods in sklearn in order to evaluate classification performance of a model

#%% model evaluation using accuracy score > the simplest method

# calculates the ratio of true predicted labels
acc_score = accuracy_score(labels, predicted_labels)
print(">>> Accuracy Score: {}".format(acc_score))

'''
as seen at the output, the accuracy score is quite high
but this metric is really enough for evaluating model performance ?

Example;
----------
you have an unbalanced dataset so that there are 100 data instances; 
    90 of them belongs to class-0, only 10 of them belongs to class-1
your model predicts the label of ALL instances, no exception, as class-0 (100 predictions and all is class-0)
in this case, the accuracy score becomes as %90 
if you only look at the accuracy score, then your model can be considered as 'successfull'
in fact, your model is not able to discriminate class-0 from class-1
it predicts true label of NONE of the instances that actually belongs to class-1
so, how this model can be considered as successfull ?
'''

#%% confusion matrix

conf_matrix = confusion_matrix(labels, predicted_labels)
print(">>> Confusion matrix: \n{}".format(conf_matrix))
'''
what is confusion matrix ?
rows corresponds to predicted class labels
cols corresponts to actual class labels

confusion matrix displays the number of 
true and false predictions for each classes
'''

#%% classification report (better method for evaluating model performance)
report = classification_report(labels, predicted_labels)
print(">>> Classification Report: \n{}".format(report))

'''
this method serves a detailed classification report;
    precision, recall and f1-score
f1-score is better than accuracy score for model evaluation
check documentations for further details
'''
