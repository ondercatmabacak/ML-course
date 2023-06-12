#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 00:26:18 2019

@author: ayf

Creating Toy Datasets for Regression
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def toFrame(x, y):
    frame = pd.DataFrame(columns=["x","y"])
    frame["x"] = x
    frame["y"] = y    
    return frame

def display2D(frame, title=""):
    # create figure
    plt.figure()
    plt.title(title, size=20)
    # scatter data set
    plt.scatter(frame["x"], frame["y"], marker="o", alpha=0.70)
    # display labels for x and y axes
    plt.xlabel("X", size=20,labelpad=15)
    plt.ylabel("Y", rotation=0, size=20,labelpad=20)
    

def save_frame(frame, name):
    fname = "{}.csv".format(name)
    frame.to_csv(fname, index=False)

def closeAll():
    plt.close("all")

def save_dataset(name, x, y):
    frame = toFrame(x, y)
    display2D(frame, name)
    save_frame(frame, name)
    return frame


def polynom(x, coeffs):
    res = 0
    for power, coeff in enumerate(reversed(coeffs)):
        res += np.power(x,power)*coeff
    return res
#%% simple linear dataset 1
# params
n_samples = 10
xmin, xmax = 1, 10
seed      = 22
noiseL, noiseH = -2, +2
coeffs = [3, 4] # 3x + 4

# conditioning randomness
np.random.seed(seed)
# create x points
x = np.random.uniform(xmin, xmax, n_samples)
# determine y points
y = polynom(x, coeffs)
# add some noise to the y
noise =  np.random.uniform(noiseL, noiseH, n_samples)
y += noise
# save data set
frame = save_dataset("linear1", x, y)
    
#%% simple linear dataset 2
# params
n_samples = 50
xmin, xmax = 1, 50
seed      = 23
noiseL, noiseH = -9, +7
coeffs = [2, 6] # 2x - 6

# conditioning randomness
np.random.seed(seed)
# create x points
x = np.random.uniform(xmin, xmax, n_samples)
# determine y points
y = polynom(x, coeffs)
# add some noise to the y
noise =  np.random.uniform(noiseL, noiseH, n_samples)
y += noise
# save data set
frame = save_dataset("linear2", x, y)

#%% simple parabola dataset 1
# params
n_samples = 10
xmin, xmax = -2, +2
seed      = 26
noiseL, noiseH = -0, +0
coeffs = [1, 0, -4] # x**2 - 4

# conditioning randomness
np.random.seed(seed)
# create x points
x = np.random.uniform(xmin, xmax, n_samples)
# determine y points
y = polynom(x, coeffs)
# add some noise to the y
noise =  np.random.uniform(noiseL, noiseH, n_samples)
y += noise
# save data set
frame = save_dataset("parabola1", x, y)

#%% simple parabola dataset 2
# params
n_samples = 50
xmin, xmax = -20, 30
seed      = 27
noiseL, noiseH = -35, +25
coeffs = [1, -8, 28] # x**2 - 8x + 28

# conditioning randomness
np.random.seed(seed)
# create x points
x = np.random.uniform(xmin, xmax, n_samples)
# determine y points
y = polynom(x, coeffs)
# add some noise to the y
noise =  np.random.uniform(noiseL, noiseH, n_samples)
y += noise
# save data set
frame = save_dataset("parabola2", x, y)



#%% simple polynomial dataset 1
# params
n_samples = 50
xmin, xmax = -3.2, +1.2
seed      = 44
noiseL, noiseH = -0, 0
coeffs = [1, 4, 1, -6, 0] 

# conditioning randomness
np.random.seed(seed)
# create x points
x = np.random.uniform(xmin, xmax, n_samples)
# determine y points
y = polynom(x, coeffs)
# add some noise to the y
noise =  np.random.uniform(noiseL, noiseH, n_samples)
y += noise
# save data set
frame = save_dataset("polynom1", x, y)

#%% simple polynomial dataset 2
# params
n_samples = 50
xmin, xmax = -5, +2
seed      = 131
noiseL, noiseH = -2, +2
coeffs = [1, 6, -2, -36, 1, 20] 

# conditioning randomness
np.random.seed(seed)
# create x points
x = np.random.uniform(xmin, xmax, n_samples)
# determine y points
y = polynom(x, coeffs)
# add some noise to the y
noise =  np.random.uniform(noiseL, noiseH, n_samples)
y += noise
# save data set
frame = save_dataset("polynom2", x, y)

#%% simple polynomial dataset 3
# params
n_samples = 100
xmin, xmax = -1.6, 1.8
seed      = 444
noiseL, noiseH = -0.5, +0.5
coeffs = [2, -1, -4, 1, -3, 3, -2, -1] 
 
# conditioning randomness
np.random.seed(seed)
# create x points
x = np.random.uniform(xmin, xmax, n_samples)
# determine y points
y = polynom(x, coeffs)
# add some noise to the y
noise =  np.random.uniform(noiseL, noiseH, n_samples)
y += noise
# save data set
frame = save_dataset("polynom3", x, y)


#%% simple complex dataset 1
n_samples = 100
xmin, xmax = -10, +10
seed      = 333
noiseL, noiseH = -0, +0

def func(x):
    return x*np.sin(x)
 
# conditioning randomness
np.random.seed(seed)
# create x points
x = np.random.uniform(xmin, xmax, n_samples)
# determine y points
y = func(x)
# add some noise to the y
noise =  np.random.uniform(noiseL, noiseH, n_samples)
y += noise
# save data set
frame = save_dataset("complex1", x, y)


#%% simple complex dataset 2
n_samples = 500
xmin, xmax = -50, +50
seed      = 333
noiseL, noiseH = -0, +0

def func(x):
    return x*np.sin(x)
 
# conditioning randomness
np.random.seed(seed)
# create x points
x = np.random.uniform(xmin, xmax, n_samples)
# determine y points
y = func(x)
# add some noise to the y
noise =  np.random.uniform(noiseL, noiseH, n_samples)
y += noise
# save data set
frame = save_dataset("complex2", x, y)


#%% simple complex dataset 2
n_samples = 20000
xmin, xmax = -50, +50
seed      = 333
noiseL, noiseH = -0, +0

def func(x):
    return x*np.sin(x)
 
# conditioning randomness
np.random.seed(seed)
# create x points
x = np.random.uniform(xmin, xmax, n_samples)
# determine y points
y = func(x)
# add some noise to the y
noise =  np.random.uniform(noiseL, noiseH, n_samples)
y += noise
# save data set
frame = save_dataset("complex3", x, y)
