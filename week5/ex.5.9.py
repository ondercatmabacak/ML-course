#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:03:25 2019

@author: ayf
"""

# import modules
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor


# invoke this method on ipython console when there are lots of plots
def closeAll():
    plt.close("all")
 

# method for reading datasets
def read_dataset(path):
    # make dataset global
    global dataset
    # load csv data set from file using pandas
    dataset = pd.read_csv(path) # the type of dataset is pandas frame 
    # check Variable explorer and see data table
    
    # we can extract data labels as follows
    outp = dataset.loc[:, "y"] # select label column
    # we can extract the actual data as follows
    inp  = dataset.loc[:, "x"] # select all columns except the label column
    # return the data and labels seperately
    return inp, outp

# method for scattering data set points
def scatter_dataset(X, Y, title=""):
    # create figure
    plt.figure()
    plt.title(title, size=20)
    # scatter data set
    plt.scatter(X, Y, marker="o", alpha=0.70)
    # display labels for x and y axes
    plt.xlabel("X", size=20,labelpad=15)
    plt.ylabel("Y", rotation=0, size=20,labelpad=20)


# method for plotting model curve
def plot_model(model, X, color=None, linestyle=None, val="center", hal="center"):
    # get min and max X values
    xmin, xmax = np.min(X), np.max(X)
    # generate a huge input vector in this range (length may be equal to 100)
    xx = np.linspace(xmin, xmax, 1000).reshape(-1,1)
    # now, estimate the output vector according to polynomial model
    yy = model.predict(xx).reshape(-1,1)
    # plot the model
    plt.plot(xx, yy, color=color, linestyle=linestyle)

    
# method for displaying a single point
def disp_point(inp, outp,color=None):
    if color is None:
        color = "black"
    plt.scatter([[inp]], [[outp]], marker="o", color=color, s=40, alpha=0.7)
    text = "({:.2f}, {:.2f})".format(inp, outp)
    plt.text(inp, outp, text, fontsize=10, 
             verticalalignment="bottom",
             horizontalalignment="left")
 

# method for calculating squared error
def squared_error(Ytrue, Ypred):
    return ((Ypred-Ytrue)**2).sum()


        
closeAll()

#%% read complex dataset

# read dataset
dname = "complex1" 
path = "./datasets/{}.csv".format(dname)
X, Y = read_dataset(path) 

# scatter dataset
scatter_dataset(X, Y, dname)

#%% create DTC regressor

# hyper params
criterion = "mse"
max_depth = None

# create a linear model (function)
reg = DecisionTreeRegressor(criterion=criterion, max_depth=max_depth, 
                            random_state=22)


#%% train model

# just data type conversion (from frame to n-dimensional array)
_X = X.values.reshape(-1,1)
_Y = Y.values.reshape(-1,1)

reg.fit(_X, _Y)

# plot curve
plot_model(reg, _X, color="red")

#%% evaluate model

# first make predictions
Ypred = reg.predict(_X)

sqerror = ((_Y-Ypred)**2).sum()
print("  > Squared error: {}".format(sqerror))
# model score (how well model fitted to the data)
score = reg.score(_X, _Y) ## metric => (1-u) / v
print("  > Model score: {}".format(score))

 
#%% KNN on complex2 dataset ?

# read dataset
dname2 = "complex2" 
path2 = "./datasets/{}.csv".format(dname2)
X2, Y2 = read_dataset(path2) 

# scatter dataset
scatter_dataset(X2, Y2, dname2)

## train model
# just data type conversion (from frame to n-dimensional array)
_X2 = X2.values.reshape(-1,1)
_Y2 = Y2.values.reshape(-1,1)
reg.fit(_X2, _Y2)

# plot curve
plot_model(reg, _X2, color="red")

## evaluate model
# first make predictions
Y2pred = reg.predict(_X2)

sqerror = ((_Y2-Y2pred)**2).sum()
print("  > Squared error: {}".format(sqerror))
# model score (how well model fitted to the data)
score = reg.score(_X2, _Y2) ## metric => (1-u) / v
print("  > Model score: {}".format(score))

