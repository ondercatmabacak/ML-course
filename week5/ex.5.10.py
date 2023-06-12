#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:18:12 2019

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
    plt.scatter(X, Y, marker="o", alpha=0.70, color="gray")
    # display labels for x and y axes
    plt.xlabel("X", size=20,labelpad=15)
    plt.ylabel("Y", rotation=0, size=20,labelpad=20)


# method for plotting model curve
def plot_model(model, X, label, linestyle=None):
    # get min and max X values
    xmin, xmax = np.min(X), np.max(X)
    # generate a huge input vector in this range (length may be equal to 100)
    xx = np.linspace(xmin, xmax, 100).reshape(-1,1)
    # now, estimate the output vector according to polynomial model
    yy = model.predict(xx).reshape(-1,1)
    # plot the model
    plt.plot(xx, yy, linestyle=linestyle, label=label)
 
# method for calculating squared error
def squared_error(Ytrue, Ypred):
    return ((Ypred-Ytrue)**2).sum()


# method for testing the model
def test_model(model, dataset_name):
    
    print("==============")
    print(dataset_name)
    
    path = "./datasets/{}.csv".format(dataset_name)
    X, Y = read_dataset(path) 
    scatter_dataset(X, Y, dataset_name)
    
    # get axes limits
    xlim, ylim = plt.xlim(), plt.ylim()
    
    # just data type conversion (from frame to n-dimensional array)
    _X = X.values.reshape(-1,1)
    _Y = Y.values.reshape(-1,1)
    
    # train model
    reg.fit(_X, _Y)
    
    # plot the model
    plot_model(reg, X, label="KNN", linestyle="--")
    
    # evaluate model
    Ypred = reg.predict(_X)
    # squared error
    sqerror = squared_error(_Y, Ypred)
    print("  > Squared error: {}".format(sqerror))
    # model score (how well model fitted to the data)
    score = reg.score(_X, _Y) ## metric => (1-u) / v
    print("  > Model score: {}".format(score))
    
    # set axes limits
    plt.xlim(xlim), plt.ylim(ylim)
    # show legends
    plt.legend(loc="best")


#%% create DTC regressor

# hyper params
criterion = "mse"
max_depth = 5

# create a linear model (function)
reg = DecisionTreeRegressor(criterion=criterion, max_depth=max_depth, 
                            random_state=22)

#%%
# closeAll()
test_model(reg, "linear1")

#%%
# closeAll()
test_model(reg, "linear2")

#%%
# closeAll()
test_model(reg, "parabola1")

#%%
# closeAll()
test_model(reg, "parabola2")

#%%
# closeAll()
test_model(reg, "polynom1")

#%%
# closeAll()
test_model(reg, "polynom2")

#%%
# closeAll()
test_model(reg, "polynom3")

#%%
# closeAll()
test_model(reg, "complex1")

#%%
# closeAll()
test_model(reg, "complex2")

