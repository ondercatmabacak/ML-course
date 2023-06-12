#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:25:08 2019

@author: ayf
"""


# import modules
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# invoke this method on ipython console when there are lots of plots
def closeAll():
    plt.close("all")

# method for generating text for a polynom
def model_to_text(coeffs):
    txt = "f(x) = "
    for pw, coef in enumerate(coeffs):
        pw = len(coeffs)-pw-1
        if pw == 0:
            txt += "{:.2f} ".format(coef)
        else:
            txt += "{:.2f}x^{} + ".format(coef,pw)
    return txt

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
def plot_model(coeffs, X, label, linestyle=None):
    # get min and max X values
    xmin, xmax = np.min(X), np.max(X)
    # generate a huge input vector in this range (length may be equal to 100)
    xx = np.linspace(xmin, xmax, 100)
    # now, estimate the output vector according to polynomial model
    yy = polynom(coeffs, xx)
    # plot the model
    plt.plot(xx, yy, linestyle=linestyle, label=label)

    
# method for displaying a single point
def disp_point(inp, outp,color=None):
    if color is None:
        color = "black"
    plt.scatter([[inp]], [[outp]], marker="o", color=color, s=40, alpha=0.7)
    text = "({:.2f}, {:.2f})".format(inp, outp)
    plt.text(inp, outp, text, fontsize=10, 
             verticalalignment="bottom",
             horizontalalignment="left")

# method for calculating value of a polynom
def polynom(coeffs, X):
    res = 0.
    for pw, coeff in enumerate(reversed(coeffs)):
        res += coeff * np.power(X, pw)
    return res

# method for calculating squared error
def squared_error(Ytrue, Ypred):
    return ((Ypred-Ytrue)**2).sum()


# method for testing the model
def test_model(X, Y, degree):
    
    print("==============")
    # get axes limits
    xlim, ylim = plt.xlim(), plt.ylim()
    
    # create a linear model (function)
    reg = LinearRegression()
    
    # just data type conversion (from frame to n-dimensional array)
    _X = X.values.reshape(-1,1)
    _Y = Y.values.reshape(-1,1)
    
    # transform input
    pf = PolynomialFeatures(degree=degree)
    _XX = pf.fit_transform(_X)
    
    # train model
    reg.fit(_XX, _Y)
    # get estimated coeffs
    coeffs = np.flip(reg.coef_.ravel()) # coeffs
    interc = reg.intercept_ # intercept
    # in this package, the intercept is not accepted as a coeff
    # so, we should add it into coeffs
    coeffs[-1] = interc     
    print(">>> The model: {}".format(model_to_text(coeffs)))
    
    # plot the model
    plot_model(coeffs, X, label=degree, linestyle="--")
    
    # evaluate model
    Ypred = reg.predict(_XX)
    # squared error
    sqerror = squared_error(_Y, Ypred)
    print("  > Squared error: {}".format(sqerror))
    # model score (how well model fitted to the data)
    score = reg.score(_XX, _Y) ## metric => (1-u) / v
    print("  > Model score: {}".format(score))
    
    # set axes limits
    plt.xlim(xlim), plt.ylim(ylim)
    # show legends
    plt.legend(loc="best")
    


    

#%% 1st dataset: polynom-1
closeAll()
# read dataset
dataset_name = "polynom1" 
path = "./datasets/{}.csv".format(dataset_name)
X, Y = read_dataset(path) 

# scatter dataset
scatter_dataset(X, Y, dataset_name)

test_model(X, Y, 1)
test_model(X, Y, 2)
test_model(X, Y, 3)
test_model(X, Y, 4) 
test_model(X, Y, 5) 

#%% 2nd dataset: polynom-2
closeAll()
# read dataset
dataset_name = "polynom2" 
path = "./datasets/{}.csv".format(dataset_name)
X, Y = read_dataset(path) 

# scatter dataset
scatter_dataset(X, Y, dataset_name)

test_model(X, Y, 1)
test_model(X, Y, 2)
test_model(X, Y, 3)
test_model(X, Y, 4) 
test_model(X, Y, 5) 
test_model(X, Y, 6) 

#%% 3rd dataset: polynom-3
closeAll()
# read dataset
dataset_name = "polynom3" 
path = "./datasets/{}.csv".format(dataset_name)
X, Y = read_dataset(path) 

# scatter dataset
scatter_dataset(X, Y, dataset_name)

# test_model(X, Y, 1)
# test_model(X, Y, 2)
# test_model(X, Y, 3)
test_model(X, Y, 4) 
test_model(X, Y, 5) 
test_model(X, Y, 6)
test_model(X, Y, 7)
test_model(X, Y, 8)

 

