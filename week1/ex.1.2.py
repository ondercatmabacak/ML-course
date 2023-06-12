#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:03:20 2019

@author: ayf

Example:
--------
Load a real data set from sklearn and visualize it
"""

# import dataset module from sklearn
from sklearn import datasets
# import pyplot module as plt (module names are commonly shortened for simplicity)
import matplotlib.pyplot as plt
# import numpy module as np (module names are commonly shortened for simplicity)
import numpy as np

# %% load data set and print details

# load digits data set (the return type is 'dict')
dataset = datasets.load_digits()

# Because the return type is dict, print dict keys ('images', 'data', 'target_names', 'DESCR', 'target')
print(">>> Dictionary Keys: %s" % dataset.keys())

# we can extract variables from dict using proper keys
# get dataset description
description = dataset["DESCR"]
# get actual data
data = dataset["data"]
# get data labels
labels = dataset["target"]
# get unique data labels
unique_labels = dataset["target_names"]
# get image array
images = dataset["images"]


# print dataset description
print(">>> Description: ")
print(description)
print("")
# print data shape (There are 1797 data points with 64 features)
print(">>> Data Shape: {}".format(data.shape))
# print data labels for each data point (if it is too much all labels are not printed to the console)
print(">>> Labels: {}".format(labels))
# print unique data labels (digits from 0 to 9)
print(">>> Unique Labels: {}".format(unique_labels))
print("")

# print how many instances for each class
group_sizes = np.bincount(labels)
print(">>> Number of intances for each label: {}".format(group_sizes))
print("")


# %% display a single data instance as an image

index = 323  # select a data index to visualize
image = images[index]  # get image array
actual_label = labels[index]  # get actual label of the image

# print the image label
print(">>> The label of {}-indexed instance is {}".format(index, actual_label))
# plot the image
plt.matshow(image, cmap=plt.cm.Greys)
_ = plt.title("{}. data vector as an image".format(index))

# %% display first 10 data points

# create a figure window with 2 rows and 5 columns
fig, axes = plt.subplots(2, 5)
# get first 10 data points
# data points from index-0 to index-10 >>> interval: [0,10)
data10 = data[0:10]
# get labels of those points
labels10 = labels[0:10]
# display each data instance
for X, label, ax in zip(data10, labels10, axes.ravel()):
    ax.set_title(label)
    ax.imshow(X.reshape(8, 8), cmap="gray_r")
    ax.set_xticks(())
    ax.set_yticks(())
plt.tight_layout()
