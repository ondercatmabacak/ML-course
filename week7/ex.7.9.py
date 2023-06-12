#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 16:15:08 2019

@author: ayf

GMM Performance on Real Dataset ?
"""

# import modules
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture

   
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



def disp_scores(result):
    # get number of cluster list
    n_cluster_list = result.keys()
    # get AIC score list
    aic_list = [result[n]["aic"] for n in result.keys()]
    # get BIC score list
    bic_list = [result[n]["bic"] for n in result.keys()]
    
    # create figure
    plt.figure()
    # get axis
    ax1 = plt.gca()
    # create second axis
    ax2 = ax1.twinx()
    
    # plot the AIC score 
    ax1.scatter(n_cluster_list, aic_list, color="tab:blue", label="AIC")
    ax1.plot(n_cluster_list, aic_list, color="tab:blue")
    # plot the score
    ax2.scatter(n_cluster_list, bic_list, color="tab:red", label="BIC")
    ax2.plot(n_cluster_list, bic_list, color="tab:red")
    
    # edit labels and other staffs
    ax1.set_xlabel("n_clusters", size=14)
    ax1.set_ylabel("Score", size=14)
    ax1.legend(loc="best")
    ax1.set_xticks(n_cluster_list)
    ax1.legend(loc='upper left')
    ax1.grid(axis="x")
    
    ax2.set_ylabel("score", size=14)
    ax2.legend(loc='upper right')

def estimate_n_clusters(data, n_cluster_set, criteria="bic"):
    # dict variable to keep created models and scores 
    result = {}
    #################
    for n_clusters in n_cluster_set:
        # create the model 
        gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", 
                              n_init=3, max_iter=1000, random_state=22)
        # train (fit) the model with data but without labels !!!
        gmm.fit(data)
        # model performance
        aic = gmm.aic(data)
        bic = gmm.bic(data)
        # print scores
        print ("  > n_cluster:  {};  AIC: {:.2f}  BIC: {: .2f}"
           .format(n_clusters, aic, bic))
        # keep the scores
        result[n_clusters] = {"model": gmm,"aic": aic, "bic": bic}
    
    #display scores
    disp_scores(result)
    
    ### determine best model
    # lowest AIC or BIC is better
    best_n_clusters = None
    min_score   = +np.inf
    
    for n_clusters in result.keys():
        # get reference score according to the specified criteria 
        score = result[n_clusters][criteria.lower()]
        if score < min_score:
            min_score = score
            best_n_clusters = n_clusters
    
    print("  > Best Number of Clusters : {}".format(best_n_clusters))
    # get best model
    kms_best = result[best_n_clusters]["model"]
    return kms_best

# method for testing model on a dataset
def test_model(data, n_clusters, criteria="bic"):
    print ("====================")
    print (">>> Testing Model")
    print ("  > n_clusters: {}".format(n_clusters))
    
    # if number of clusters type is list, then estimate the best value
    if isinstance(n_clusters, list):
        model = estimate_n_clusters(data, n_cluster_set=n_clusters, 
                                    criteria=criteria)
        n_clusters = model.n_components
    elif isinstance(n_clusters, int):
        model = GaussianMixture(n_components=n_clusters, covariance_type="full", 
                                n_init=3, max_iter=1000, random_state=22)
        model.fit(data)
    else:
        raise Exception("invalid n_clusters")
  
    # evaluate model performance (unsupervised)
    aic_score = model.aic(data)
    bic_score = model.bic(data)
    print("  > AIC Score: {}".format(aic_score))
    print("  > BIC Score: {}".format(bic_score))
    
##################################################
################ TOY DATASETS  ###################
##################################################
    
#%% iris data set
closeAll()
# read data set
dataset_name = "iris"
path = "../datasets/{}/{}.csv".format(dataset_name,dataset_name)
print("\n::::: {} :::::\n".format(dataset_name))
data, true_labels = read_dataset(path, "species")

# test model (estimate number of clusters)
test_model(data, n_clusters=range(2, 10), criteria="bic")
 

#%% digits data set
closeAll()
# read data set
dataset_name = "optdigits"
path = "../datasets/{}/{}.csv".format(dataset_name,dataset_name)
print("\n::::: {} :::::\n".format(dataset_name))
data, true_labels = read_dataset(path, "digit")

# test model (estimate number of clusters)
test_model(data, n_clusters=range(5, 21), criteria="bic")


#%% wdbc data set
closeAll()
# read data set
dataset_name = "wdbc"
path = "../datasets/{}/{}.csv".format(dataset_name,dataset_name)
print("\n::::: {} :::::\n".format(dataset_name))
data, true_labels = read_dataset(path, "diagnosis")

# test model (estimate number of clusters)
test_model(data, n_clusters=range(2, 7), criteria="bic")


#%% htru2 data set
closeAll()
# read data set
dataset_name = "htru2"
path = "../datasets/{}/{}.csv".format(dataset_name,dataset_name)
print("\n::::: {} :::::\n".format(dataset_name))
data, true_labels = read_dataset(path, "ispulsar")

# test model (estimate number of clusters)
test_model(data, n_clusters=range(2, 10), criteria="bic")
