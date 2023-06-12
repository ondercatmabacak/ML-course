#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 00:25:51 2019

@author: ayf

Linear Regression
"""


# import modules
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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
def plot_model(a, b, X, color=None, linestyle=None, val="center", hal="center"):
    # get min and max X values
    xmin, xmax = np.min(X), np.max(X)
    # generate a huge input vector in this range (length may be equal to 100)
    xx = np.linspace(xmin, xmax, 100)
    # now, estimate the output vector according to our linear model
    yy = a*xx + b
    # plot the model
    plt.plot(xx, yy, color=color, linestyle=linestyle)
    # print model text
    model_text = "f(x) = {:.2f}x + {:.2f}".format(a, b)
    xmed, ymed = np.median(xx), np.median(yy)
    plt.text(xmed, ymed, model_text,
             verticalalignment=val,
             horizontalalignment=hal)
    
# method for displaying a single point
def disp_point(inp, outp,color=None):
    if color is None:
        color = "black"
    plt.scatter([[inp]], [[outp]], marker="o", color=color, s=40, alpha=0.7)
    text = "({:.2f}, {:.2f})".format(inp, outp)
    plt.text(inp, outp, text, fontsize=10, 
             verticalalignment="bottom",
             horizontalalignment="left")
 
        
closeAll()

#%% read simple linear dataset
title =  "linear1"
path = "./datasets/{}.csv".format(title)

X, Y = read_dataset(path) 

# scatter dataset
scatter_dataset(X, Y, "Data Set")

#%% problem definition

# define a new input
inp = 8

# for a novel input "8", what is the best output ? 
# plot a vertical line for x = 8
plt.axvline(inp, color="gray", linestyle="--")
disp_point(inp, outp=20) # ??
disp_point(inp, outp=25) # ??
disp_point(inp, outp=30) # ??

#%% create linear models (functions)
# linear model: f(x) = a*x+b
# "a" is called as "slope"
# "b" is called as "intercept"
# how can we evaluate if a model is good or bad ?
# how can we estimate best "a" and "b" ?

# scatter data points
scatter_dataset(X, Y)

## which model is better ?
# f(x) = x-10
a, b = 1, 10
plot_model(a, b, X, color="green", linestyle="--", val="top", hal="left")
# calculate squared error
Ypred = a*X + b
error = ((Y-Ypred)**2).sum()
print(">>> The model: f(x) = {:.2f}x + {:.2f}".format(a, b))
print("  > Error: {}".format(error))
print("")

# f(x) = 3x+5
a, b = 3, 5
plot_model(a, b, X, color="red", linestyle="--", val="bottom", hal="right")
Ypred = a*X + b
error = ((Y-Ypred)**2).sum()
print(">>> The model: f(x) = {:.2f}x + {:.2f}".format(a, b))
print("  > Error: {}".format(error))
print("")



#%% estimating best linear model (function)  
# cost function: squared error
# calculate parameters (a and b) that minimize squared error
N = X.shape[0] # number of instances
sumY = Y.sum() 
sumX = X.sum()
sumXY = (X*Y).sum()
sumX2 = (X**2).sum()

# formula for slope
a = (N*sumXY - sumX*sumY)/(N*sumX2 - sumX**2)
# formula for intercept
b = (sumY*sumX2 - sumX*sumXY)/(N*sumX2 - sumX**2)

# thats it, we found the model;
print(">>> Best model: f(x) = {:.2f}x + {:.2f}".format(a, b))
# this model is the best one that fits the dataset
# plot the estimated linear model
plot_model(a, b, X, color="black", linestyle="solid", val="top", hal="left")
Ypred = a*X + b
error = ((Y-Ypred)**2).sum()
print(">>> The model: f(x) = {:.2f}x + {:.2f}".format(a, b))
print("  > Error: {}".format(error))
print


#%% now, estimate the best output for the novel input=8
scatter_dataset(X, Y)
# plot a vertical line for x = 8
plt.axvline(inp, color="gray", linestyle="--")
disp_point(inp, outp=20) # ??
disp_point(inp, outp=25) # ??
disp_point(inp, outp=30) # ??

# plot the estimated linear model
plot_model(a, b, X, color="black", linestyle="solid", val="top", hal="left")

outp = a*inp+b
print(">>> f({})={:.2f}".format(inp,outp))
disp_point(inp, outp, color="red")

