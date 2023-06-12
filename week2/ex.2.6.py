#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 04:56:22 2019

@author: ayf

Example:
-------
The 1st Classification Algorithm: K-Nearest Neighbor

KNN performance on real datasets
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
path = "../datasets/iris/iris.csv"

print("::::: Iris Dataset :::::\n")
data, labels = read_dataset(path, "species")

knn.fit(data, labels)
preds = knn.predict(data)

acc_score = accuracy_score(labels, preds)
print(">>> Accuracy Score: {}".format(acc_score))
conf_matrix = confusion_matrix(labels, preds)
print(">>> Confusion matrix: \n{}".format(conf_matrix))
report = classification_report(labels, preds)
print(">>> Classification Report: \n{}".format(report))
print("")

#%% optical digits dataset (training)
path = "../datasets/optdigits/optdigits.csv"

print("::::: Optical Digits Dataset  :::::\n")
data, labels = read_dataset(path, "digit")

knn.fit(data, labels)
preds = knn.predict(data)

acc_score = accuracy_score(labels, preds)
print(">>> Accuracy Score: {}".format(acc_score))
conf_matrix = confusion_matrix(labels, preds)
print(">>> Confusion matrix: \n{}".format(conf_matrix))
report = classification_report(labels, preds)
print(">>> Classification Report: \n{}".format(report))
print("")


#%% breast cancer dataset (Wisconsin Diagnostic Breast Cancer)
path = "../datasets/wdbc/wdbc.csv"

print("::::: Breast Dataset :::::\n")
data, labels = read_dataset(path, "diagnosis")

knn.fit(data, labels)
preds = knn.predict(data)

acc_score = accuracy_score(labels, preds)
print(">>> Accuracy Score: {}".format(acc_score))
conf_matrix = confusion_matrix(labels, preds)
print(">>> Confusion matrix: \n{}".format(conf_matrix))
report = classification_report(labels, preds)
print(">>> Classification Report: \n{}".format(report))
print("")

#%% htru2 dataset (High Time Resolution Universe Collaboration using the Parkes Observatory)
path = "../datasets/htru2/htru2.csv"

print("::::: HTRU2 Dataset :::::\n")
data, labels = read_dataset(path, "ispulsar")

knn.fit(data, labels)
preds = knn.predict(data)

acc_score = accuracy_score(labels, preds)
print(">>> Accuracy Score: {}".format(acc_score))
conf_matrix = confusion_matrix(labels, preds)
print(">>> Confusion matrix: \n{}".format(conf_matrix))
report = classification_report(labels, preds)
print(">>> Classification Report: \n{}".format(report))
print("")