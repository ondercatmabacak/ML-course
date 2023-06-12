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
# import pyplot module as plt (module names are commonly shortened for simplicity)
# import numpy module as np (module names are commonly shortened for simplicity)

# %% create a dataset with 2 nested circles and visualize it
%%
# 'make_circles' method generates data set  with the shape of 2 nested circles
# check documentation for further information
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
# some parameters for 'make_circles' method
n_samples = 100  # how many data points will be generated ?
noise = 0.  # std for Gaussian noise (if no noise, leave it as 0)
# change it to another integer to obtain a different data set (if None, the result will be different for each run)
seed = 22
factor = 0.5  # scale factor between inner and outer circle.


# create a toy data set with parameters defined above
dset = datasets.make_circles(
    n_samples=n_samples, noise=noise, random_state=seed, factor=factor)

# extract actual data set and target values (a.k.a data labels)
data = dset[0]
labels = dset[1]

# print data and labels (you can also see values in Variable explorer)
print(">>> Actual Data Set: ")
print(data)  # it is a matrix with n_samples row and 2 columns
print("")

print(">>> Data Labels: ")
print(labels)  # it is a vector with n_samples size
print("")

# print unique labels for data set
unique_labels = np.unique(labels)
print(">>> Unique Data Labels: ")
print(unique_labels)
print("")

# display the data set using pyplot
# create a figure window
fig = plt.figure()
# set the title
plt.title("Data Set with 2 Circles")
# scatter each distinct data group with different color & marker
# data instances belonging to the 0-labeled group
# 1st column of data matrix corresponds x values
x_points_for_label_0 = data[labels == 0, 0]
# 2nd column of data matrix corresponds y values
y_points_for_label_0 = data[labels == 0, 1]
plt.scatter(x_points_for_label_0, y_points_for_label_0,
            marker="*", color="red", label="Class-0")
# data instances belonging to the 1-labeled group
# 1st column of data matrix corresponds x values
x_points_for_label_1 = data[labels == 1, 0]
# 2nd column of data matrix corresponds y values
y_points_for_label_1 = data[labels == 1, 1]
plt.scatter(x_points_for_label_1, y_points_for_label_1,
            marker="o", color="blue", label="Class-1")
# invoke the following method to display class labels
plt.legend()

# change parameters and see different data sets
# for example, increase noise varaible in order to disrupt smooth circles
# or increase factor variable to decrase gap between circles


# %% create another data set using 'make_classification' method

# 'make_classification' method allows to generate multi-labeled & multi-dimensional data set
# check documentation for further information

# some parameters for 'make_classification' method
n_samples = 100  # how many data points will be generated ?
n_classes = 3  # how many different group will be in data set
# change it to another integer to obtain a different data set (if None, the result will be different for each run)
seed = None
# larger values spread out the clusters/classes and make the classification task easier.
class_sep = 3.0

dset = datasets.make_classification(n_samples=n_samples,
                                    # how many feature (or dimension)  (do not change it !)
                                    n_features=2,
                                    n_classes=n_classes,
                                    random_state=seed,
                                    n_informative=2,
                                    n_redundant=0,
                                    n_clusters_per_class=1, class_sep=class_sep)

# extract actual data set and target values (a.k.a data labels)
data = dset[0]
labels = dset[1]

# print data and labels (you can also see values in Variable explorer)
print(">>> Actual Data Set: ")
print(data)  # it is a matrix with n_samples row and 2 columns
print("")

print(">>> Data Labels: ")
print(labels)  # it is a vector with n_samples size
print("")

# print unique labels for data set
unique_labels = np.unique(labels)
print(">>> Unique Data Labels: ")
print(unique_labels)
print("")

# display the data set using pyplot
# create a figure window
fig = plt.figure()
# set the title
plt.title("Data Set with 2 Circles")
# scatter each distinct data group with different color
for label in unique_labels:
    # 1st column of data matrix corresponds x values
    x_points = data[labels == label, 0]
    # 2nd column of data matrix corresponds y values
    y_points = data[labels == label, 1]
    label_text = "class-{}".format(label)
    plt.scatter(x_points, y_points, label=label_text)

# invoke the following method to display class labels
plt.legend()

# change parameters and observe different data sets
# for example, decrease class_sep varaible in order to make classification problem difficult
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:19:15 2019

@author: ayf

Example:
--------
Create a toy data set and visualize it
"""

# import dataset module from sklearn
# import pyplot module as plt (module names are commonly shortened for simplicity)
# import numpy module as np (module names are commonly shortened for simplicity)

# %% create a dataset with 2 nested circles and visualize it

# 'make_circles' method generates data set  with the shape of 2 nested circles
# check documentation for further information

# some parameters for 'make_circles' method
n_samples = 100  # how many data points will be generated ?
noise = 0.  # std for Gaussian noise (if no noise, leave it as 0)
# change it to another integer to obtain a different data set (if None, the result will be different for each run)
seed = 22
factor = 0.5  # scale factor between inner and outer circle.

# create a toy data set with parameters defined above
dset = datasets.make_circles(
    n_samples=n_samples, noise=noise, random_state=seed, factor=factor)

# extract actual data set and target values (a.k.a data labels)
data = dset[0]
labels = dset[1]

# print data and labels (you can also see values in Variable explorer)
print(">>> Actual Data Set: ")
print(data)  # it is a matrix with n_samples row and 2 columns
print("")

print(">>> Data Labels: ")
print(labels)  # it is a vector with n_samples size
print("")

# print unique labels for data set
unique_labels = np.unique(labels)
print(">>> Unique Data Labels: ")
print(unique_labels)
print("")

# display the data set using pyplot
# create a figure window
fig = plt.figure()
# set the title
plt.title("Data Set with 2 Circles")
# scatter each distinct data group with different color & marker
# data instances belonging to the 0-labeled group
# 1st column of data matrix corresponds x values
x_points_for_label_0 = data[labels == 0, 0]
# 2nd column of data matrix corresponds y values
y_points_for_label_0 = data[labels == 0, 1]
plt.scatter(x_points_for_label_0, y_points_for_label_0,
            marker="*", color="red", label="Class-0")
# data instances belonging to the 1-labeled group
# 1st column of data matrix corresponds x values
x_points_for_label_1 = data[labels == 1, 0]
# 2nd column of data matrix corresponds y values
y_points_for_label_1 = data[labels == 1, 1]
plt.scatter(x_points_for_label_1, y_points_for_label_1,
            marker="o", color="blue", label="Class-1")
# invoke the following method to display class labels
plt.legend()

# change parameters and see different data sets
# for example, increase noise varaible in order to disrupt smooth circles
# or increase factor variable to decrase gap between circles


# %% create another data set using 'make_classification' method

# 'make_classification' method allows to generate multi-labeled & multi-dimensional data set
# check documentation for further information

# some parameters for 'make_classification' method
n_samples = 100  # how many data points will be generated ?
n_classes = 3  # how many different group will be in data set
# change it to another integer to obtain a different data set (if None, the result will be different for each run)
seed = None
# larger values spread out the clusters/classes and make the classification task easier.
class_sep = 3.0

dset = datasets.make_classification(n_samples=n_samples,
                                    # how many feature (or dimension)  (do not change it !)
                                    n_features=2,
                                    n_classes=n_classes,
                                    random_state=seed,
                                    n_informative=2,
                                    n_redundant=0,
                                    n_clusters_per_class=1, class_sep=class_sep)

# extract actual data set and target values (a.k.a data labels)
data = dset[0]
labels = dset[1]

# print data and labels (you can also see values in Variable explorer)
print(">>> Actual Data Set: ")
print(data)  # it is a matrix with n_samples row and 2 columns
print("")

print(">>> Data Labels: ")
print(labels)  # it is a vector with n_samples size
print("")

# print unique labels for data set
unique_labels = np.unique(labels)
print(">>> Unique Data Labels: ")
print(unique_labels)
print("")

# display the data set using pyplot
# create a figure window
fig = plt.figure()
# set the title
plt.title("Data Set with 2 Circles")
# scatter each distinct data group with different color
for label in unique_labels:
    # 1st column of data matrix corresponds x values
    x_points = data[labels == label, 0]
    # 2nd column of data matrix corresponds y values
    y_points = data[labels == label, 1]
    label_text = "class-{}".format(label)
    plt.scatter(x_points, y_points, label=label_text)

# invoke the following method to display class labels
plt.legend()

# change parameters and observe different data sets
# for example, decrease class_sep varaible in order to make classification problem difficult

# %%
