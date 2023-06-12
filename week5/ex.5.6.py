#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:12:53 2019

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
def test_model(model, X, Y, degree):
    
    print("==============")
    # get axes limits
    xlim, ylim = plt.xlim(), plt.ylim()
    
    # just data type conversion (from frame to n-dimensional array)
    _X = X.values.reshape(-1,1)
    _Y = Y.values.reshape(-1,1)
    
    # transform input
    pf = PolynomialFeatures(degree=degree)
    _XX = pf.fit_transform(_X)
    
    # train model
    model.fit(_XX, _Y)
    # get estimated coeffs
    coeffs = np.flip(model.coef_.ravel()) # coeffs
    interc = model.intercept_ # intercept
    # in this package, the intercept is not accepted as a coeff
    # so, we should add it into coeffs
    coeffs[-1] = interc     
    print(">>> The model: {}".format(model_to_text(coeffs)))
    
    # plot the model
    plot_model(coeffs, X, label=degree, linestyle="--")
    
    # evaluate model
    Ypred = model.predict(_XX)
    # squared error
    sqerror = ((_Y-Ypred)**2).sum()
    print("  > Squared error: {}".format(sqerror))
    # model score (how well model fitted to the data)
    score = model.score(_XX, _Y) ## metric => (1-u) / v
    print("  > Model score: {}".format(score))
    
    # set axes limits
    plt.xlim(xlim), plt.ylim(ylim)
    # show legends
    plt.legend(loc="best")
    
    return coeffs
    

#%% 1st dataset: polynom-1
closeAll()
# read dataset
dname1 = "complex1" 
path1 = "./datasets/{}.csv".format(dname1)
X1, Y1 = read_dataset(path1) 

# scatter dataset
scatter_dataset(X1, Y1, dname1)

# create a linear model (function)
reg1 = LinearRegression()
# test model with different degrees
for degree in range(6, 16):
    test_model(reg1, X1, Y1, degree)
    plt.pause(0.25)
    plt.draw()
    raw_input("Press enter to continue")

best_degree_1 = 14
scatter_dataset(X1, Y1, best_degree_1)
test_model(reg1, X1, Y1, degree=14)


#%% 2nd dataset: polynom-2
dname2 = "complex2" 
path2 = "./datasets/{}.csv".format(dname2)
X2, Y2 = read_dataset(path2) 

# scatter dataset
scatter_dataset(X2, Y2, dname2)
# test same model on complex2 dataset
test_model(reg1, X2, Y2, degree=14)


#%% test

# scatter dataset
scatter_dataset(X2, Y2, dname2)
# create a linear model (function)
reg2 = LinearRegression()
# test model with different degrees
for degree in np.arange(10, 40, step=5):
    test_model(reg2, X2, Y2, degree)
    # plt.pause(0.25)
    plt.draw()
    # raw_input("Press enter to continue")
 
