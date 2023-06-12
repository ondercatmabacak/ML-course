#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:36:31 2019

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
    xx = np.linspace(xmin, xmax, 100)
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
title =  "parabola1"
path = "./datasets/{}.csv".format(title)

X, Y = read_dataset(path) 

# scatter dataset
scatter_dataset(X, Y, "Data Set")

#%% problem definition

# define a new input
inp = 0.5

# for a novel input "8", what is the best output ? 
# plot a vertical line for x = 0.5
plt.axvline(inp, color="gray", linestyle="--")
disp_point(inp, outp=-4.0) # ??
disp_point(inp, outp=-3.5) # ??
disp_point(inp, outp=-3.0) # ??

#%% create models (functions)

# scatter data points
scatter_dataset(X, Y)
# get axes limits
_xlim, _ylim = plt.xlim(), plt.ylim()

### which model is better ?
### a linear model or a parabolic model

# linear model ?
reg = LinearRegression()
reg.fit(X.values.reshape(-1,1), Y.values.reshape(-1,1))
a, b = reg.coef_[0,0], reg.intercept_[0]
Ypred = polynom([a,b], X)
error = squared_error(Y, Ypred)
print(">>> The model: {}".format(model_to_text([a,b])))
print("  > Error: {}".format(error))
print("")
plot_model([a,b], X, color="green", linestyle="--", val="top", hal="left")
plt.xlim(_xlim), plt.ylim(_ylim) # fix axes limits

# parabolic model: f(x) = a*x^2 +b*x + c
# "a", "b" are coefficients
# "c" is "intercept"
a, b, c = 1, +1, -3.5
plot_model([a, b, c], X, color="red", linestyle="--", val="bottom", hal="right")
plt.xlim(_xlim), plt.ylim(_ylim) # fix axes limits
# calculate squared error
Ypred = polynom([a,b,c], X)
error = squared_error(Y,Ypred)
print(">>> The model: {}".format(model_to_text([a,b,c])))
print("  > Error: {}".format(error))
print("")

# are there a better model ?

#%% learning best parabolic model (function)  
# cost function: squared error
# calculate coeffs that minimize squared error using linear algebra 

# just data type conversion (from frame to n-dimensional array)
_X = X.values.reshape(-1,1)
_Y = Y.values.reshape(-1,1)

# we need polynomial features (with degree=2 for parabola)

pf = PolynomialFeatures(degree=2)
_XX = pf.fit_transform(_X)

# calculate coefficients using linear algebra
step1 = np.dot(_XX.transpose(), _XX)
step2 = np.linalg.inv(step1)
step3 = np.dot(step2, _XX.transpose())
step4 = np.dot(step3, _Y)
# thats it !
coeffs  = np.flip(step4.reshape(1,-1))[0]
# we found the model;
print(">>> The model: {}".format(model_to_text(coeffs)))

# plot the estimated linear model
plot_model(coeffs, X, color="black", linestyle="solid", val="top", hal="left")
# calculate squared error
Ypred = polynom(coeffs, X)
error = squared_error(Y,Ypred)
print(">>> The model: {}".format(model_to_text(coeffs)))
print("  > Error: {}".format(error))
print("")


#%% now, estimate the best output for the novel input=8
scatter_dataset(X, Y)
# plot a vertical line for x = 0.5
plt.axvline(inp, color="gray", linestyle="--")
disp_point(inp, outp=-4.0) # ??
disp_point(inp, outp=-3.5) # ??
disp_point(inp, outp=-3.0) # ??

# plot the estimated linear model
plot_model(coeffs, X, color="black", linestyle="solid", val="bottom", hal="right")

outp = polynom(coeffs,inp)
print(">>> f({})={:.2f}".format(inp,outp))
disp_point(inp, outp, color="red")

