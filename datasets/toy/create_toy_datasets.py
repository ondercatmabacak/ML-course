#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 02:50:44 2019

@author: ayf

---
Creating toy datasets
"""

import os
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd

def toFrame(data, labels):
    frame = pd.DataFrame()
    for i in xrange(data.shape[1]):
        col = "f{}".format(i)
        frame.loc[:, col] = data[:,i]
    frame.loc[:, "label"] = labels
    return frame

def display2D(frame, title):
    plt.figure()
    plt.title(title, size=20)
    groups = frame.groupby("label").groups
    for label in groups.keys():
        indices = groups[label]
        subframe = frame.loc[indices]
        xpoints = subframe.iloc[:,0]
        ypoints = subframe.iloc[:,1]
        label_text = "class: {}".format(label)
        plt.scatter(xpoints, ypoints, marker="o", alpha=0.70, label=label_text)
    # place legend box at the best location
    plt.legend(loc="best", fontsize=14)
    # display labels for x and y axes
    plt.xlabel("f1", size=14)
    plt.ylabel("f2", rotation=0, size=14)
    

def save_frame(frame, name):
    fname = "{}.csv".format(name)
    frame.to_csv(fname, index=False)

def closeAll():
    plt.close("all")

def save_dataset(name, data, labels, save_image=True):
    frame = toFrame(data, labels)
    display2D(frame, name)
    save_frame(frame, name)
    return frame


#%% intermixed dataset
def create_dataset(n_samples, n_classes, hypercube, class_sep, random_state, name, save_image=True):
    
    data, labels = datasets.make_classification(n_samples=n_samples, n_classes=n_classes,
                                                n_features=2, n_redundant=0, n_informative=2,
                                                 n_clusters_per_class=1, class_sep=class_sep,
                                                 hypercube=hypercube, random_state=random_state)
    return save_dataset(name, data, labels)

frame = create_dataset(200, 2, False, 1.8, 222, "binary_intermixed")
frame = create_dataset(300, 3, False, 3, 123, "multilabel_intermixed")

    
#%% blobs dataset
def create_blob_dataset(n_samples, n_classes, cluster_std, random_state, name):
    
    data, labels = datasets.make_blobs(n_samples=n_samples, n_features=2, 
                                   centers=n_classes, cluster_std=cluster_std, 
                                   random_state=random_state)
    return save_dataset(name, data, labels)

frame = create_blob_dataset(200, 2, 1.0, 22, "binary_blobs")
frame = create_blob_dataset(200, 2, 2.0, 22, "binary_blobs2")
frame = create_blob_dataset(300, 3, 1.0, 23, "multilabel_blobs")

#%% moon dataset
def create_moon_dataset(n_samples, noise, random_state, name):
    
    data, labels = datasets.make_moons(n_samples=n_samples, noise=noise, 
                                       random_state=random_state) 
    return save_dataset(name, data, labels)

frame = create_moon_dataset(200, 0.1, 22, "binary_moons")

#%% circle dataset
def create_circle_dataset(n_samples, noise, factor, random_state, name):
    
    data, labels = datasets.make_circles(n_samples=n_samples, noise=noise, 
                                         factor=factor, random_state=random_state)
    return save_dataset(name, data, labels)

frame = create_circle_dataset(200, 0.1, 0.4, 22, "binary_circles")



