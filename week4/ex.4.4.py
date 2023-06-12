#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:54:10 2019

@author: ayf

Example:
-------
The 3rd Classification algorithm: Naive Bayes Classifier

Model performance on real datasets
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
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


closeAll()

#%% create a classification model: Gaussian Naive Bayes (Gaussian NB)
gnb = GaussianNB()

##################################################
################ REAL DATASETS  ##################
##################################################
#%% iris dataset
path = "../datasets/iris/iris.csv"

print("::::: Iris Dataset :::::\n")
data, labels = read_dataset(path, "species")

# train model
gnb.fit(data, labels)
# make predictions
preds = gnb.predict(data)

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

# train model
gnb.fit(data, labels)
# make predictions
preds = gnb.predict(data)

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

# train model
gnb.fit(data, labels)
# make predictions
preds = gnb.predict(data)

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

# train model
gnb.fit(data, labels)
# make predictions
preds = gnb.predict(data)

acc_score = accuracy_score(labels, preds)
print(">>> Accuracy Score: {}".format(acc_score))
conf_matrix = confusion_matrix(labels, preds)
print(">>> Confusion matrix: \n{}".format(conf_matrix))
report = classification_report(labels, preds)
print(">>> Classification Report: \n{}".format(report))
print("")
 