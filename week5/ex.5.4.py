#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:11:58 2019

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
    plt.scatter(X, Y, marker="o", alpha=0.70)
    # display labels for x and y axes
    plt.xlabel("X", size=20,labelpad=15)
    plt.ylabel("Y", rotation=0, size=20,labelpad=20)


# method for plotting model curve
def plot_model(coeffs, X, color=None, linestyle=None, val="center", hal="center"):
    # get min and max X values
    xmin, xmax = np.min(X), np.max(X)
    # generate a huge input vector in this range (length may be equal to 100)
    xx = np.linspace(xmin, xmax, 10)
    # now, estimate the output vector according to polynomial model
    yy = polynom(coeffs, xx)
    # plot the model
    plt.plot(xx, yy, color=color, linestyle=linestyle)
    # print model text
    model_text = model_to_text(coeffs)
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

# method for calculating value of a polynom
def polynom(coeffs, X):
    res = 0.
    for pw, coeff in enumerate(reversed(coeffs)):
        res += coeff * np.power(X, pw)
    return res

# method for calculating squared error
def squared_error(Ytrue, Ypred):
    return ((Ypred-Ytrue)**2).sum()


        
closeAll()

#%% read simple parabola dataset
title =  "parabola2"
path = "./datasets/{}.csv".format(title)

X, Y = read_dataset(path) 

# scatter dataset
scatter_dataset(X, Y, "Data Set")

#%% problem definition

# define a new input
inp = -7
# for a novel input "8", what is the best output ? 
# plot a vertical line for x = 0.5
plt.axvline(inp, color="gray", linestyle="--")

#%% create a linear model (function)
reg = LinearRegression()

# just data type conversion (from frame to n-dimensional array)
_X = X.values.reshape(-1,1)
_Y = Y.values.reshape(-1,1)

# we need polynomial features (with degree=2 for parabola)
pf = PolynomialFeatures(degree=2)
_XX = pf.fit_transform(_X)

# train the model
reg.fit(_XX, Y)

#%% examine the estimated parameters

# get estimated coeffs
coeffs = np.flip(reg.coef_) # coeffs
interc = reg.intercept_ # intercept

# in this package, the intercept is not accepted as a coeff
# so, we should add it into coeffs
coeffs[-1] = interc
# print the model
print(">>> The model: {}".format(model_to_text(coeffs)))
 
#%% display model
plot_model(coeffs, X, color="black", val="bottom",hal="left")


#%% model evaluation 
# make prediction for training set
Ypred = reg.predict(_XX)
# squared error
sqerror = ((_Y-Ypred)**2).sum()
print(">>> Squared error: {}".format(sqerror))
# model score (how well model fitted to the data)
score = reg.score(_XX, _Y) ## metric => (1-u) / v
print(">>> Model score: {}".format(score))

#%% testing model
# estimate value of model for inp=-7
outp = polynom(coeffs, inp)
print ">>> f({})={:.2f}".format(inp,outp)
disp_point(inp, outp, color="red")

 
