#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 07:33:49 2019

@author: ayf

Example:
-------
The 2nd Classification Algorithm: Decision Tree Classifier

Understanding effects of hyper-parameters on model (binary moons dataset)
"""


# import modules
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
   
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
    for label in sorted(labels.unique()):
        # get data points belonging to the group with 'label' labeled
        label_points = data[labels==label].values
        # get x and y points
        x_points = label_points[:,0]
        y_points = label_points[:,1]
        # determine label text in figure legend
        label_text = "class-{}".format(label)
        # scatter datapoints belonging to the 'label'
        plt.scatter(x_points, y_points, marker="o", alpha=0.70, label=label_text)
    
    # place legend box at the best location
    plt.legend(loc="best", fontsize=14)
    # display labels for x and y axes
    plt.xlabel(col1, size=14)
    plt.ylabel(col2, rotation=0, size=14)


# method for displaying a single point
def disp_point(point):
   plt.scatter(point[:,0], point[:,1], marker="o", color="black", s=60, alpha=0.7)
   plt.text(point[0,0], point[0,1], "?", fontsize=15, 
            verticalalignment="bottom",
            horizontalalignment="left")

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
    # plot grid points with their predicted labels that corresponds to a color
    plt.imshow(Z, interpolation="gaussian", zorder=-100, 
               alpha=0.3, extent=[x_min, x_max, y_min, y_max], 
               aspect="auto", origin="lower") 
        

# plot and save tree
def plot_tree(model, data, labels, suffix=""):
    #from IPython.display import Image  
    import pydotplus
    import io 
    from matplotlib import pyplot as plt
    import matplotlib.image as mpimg
    
    feature_names = data.columns
    class_names = [str(i) for i in labels.unique()]
    # Create DOT data
    dot_data = export_graphviz(model, feature_names=feature_names, class_names=class_names, filled=True)
    # Draw graph
    graph = pydotplus.graph_from_dot_data(dot_data)  
    # save graph to a png file
    file_name = "tree_{}.png".format(suffix)
    graph.write_png(file_name)
    
    # Display graph on matplotlib figure 
    png = graph.create_png()
    byts = io.BytesIO(png)
    byts = mpimg.imread(byts, format='PNG')
    plt.figure()
    plt.imshow(byts, interpolation='nearest')
    plt.show()
    


#%% load blobs dataset and visualize
title =  "binary_moons"
path = "../datasets/toy/{}.csv".format(title)

data, labels = read_dataset(path, "label")
# display the scatter plot for this data set
# plot_dataset(data, "f0", "f1", labels, title) 

#%% create a classification model: DTC (with various max_depth values)
closeAll()
criterion = "entropy"
max_depth_values = xrange(2, 7) # various max_depth values as a sequence from 2 to 6
for max_depth in max_depth_values:
    dtc = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=22)
    # train (fit) the model with dataset
    dtc.fit(data, labels)
    # plot boundary
    plot_dataset(data, "f0", "f1", labels, title) 
    plt.title("max_depth: {}".format(max_depth), size=20)
    plot_boundary(dtc)
    # plot tree
    plot_tree(dtc, data, labels, suffix=max_depth)
    plt.draw()
    # plt.pause(1)
    # raw_input("Press enter to continue...")
 
#%% create a classification model: DTC (with various criteria values)
closeAll()
max_depth  = 7 
criterions = ["gini", "entropy"]
for criterion in criterions:
    dtc = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=22)
    # train (fit) the model with dataset
    dtc.fit(data, labels)
    # plot boundary
    plot_dataset(data, "f0", "f1", labels, title) 
    plt.title("criterion: {}".format(criterion), size=20)
    plot_boundary(dtc)
    # plot tree
    plot_tree(dtc, data, labels, suffix=criterion)
    plt.draw()
    # plt.pause(1)
    # raw_input("Press enter to continue...")
    




