
import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs,make_circles,make_moons
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage 
import math
from sklearn.cluster import AgglomerativeClustering
import pickle


# ----------------------------------------------------------------------
"""
Part 1: 
Evaluation of k-Means over Diverse Datasets: 
In the first task, you will explore how k-Means perform on datasets with diverse structure.
"""


def fit_kmeans(data_set, n_clusters):
    scaler = StandardScaler()
    data_set = scaler.fit_transform(data_set)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='random')
    kmeans.fit(data_set)
   
    return kmeans.labels_


def compute():
    answers = {}

    """
    A.	Load the following 5 datasets with 100 samples each: noisy_circles (nc), noisy_moons (nm), blobs with varied variances (bvv), Anisotropicly distributed data (add), blobs (b). Use the parameters from (https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html), with any random state. (with random_state = 42). Not setting the correct random_state will prevent me from checking your results.
    """
    n_samples=100
    r_state=42
    nc = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05, random_state=r_state)
    nm = datasets.make_moons(n_samples=n_samples, noise=.05, random_state=r_state)
    b = datasets.make_blobs(n_samples=n_samples, random_state=r_state)
    bvv= datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=r_state)
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=r_state)
    transf = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transf)
    add = (X_aniso, y)

    dct = answers["1A: datasets"] = {'nc': [nc[0],nc[1]],
                                     'nm': [nm[0],nm[1]],
                                     'bvv': [bvv[0],bvv[1]],
                                     'add': [add[0],add[1]],
                                     'b': [b[0],b[1]]}

    """
   B. Write a function called fit_kmeans that takes dataset (before any processing on it), i.e., pair of (data, label) Numpy arrays, and the number of clusters as arguments, and returns the predicted labels from k-means clustering. Use the init='random' argument and make sure to standardize the data (see StandardScaler transform), prior to fitting the KMeans estimator. This is the function you will use in the following questions. 
    """

    dct = answers["1B: fit_kmeans"] = fit_kmeans
    results = dct


    """
    C.	Make a big figure (4 rows x 5 columns) of scatter plots (where points are colored by predicted label) with each column corresponding to the datasets generated in part 1.A, and each row being k=[2,3,5,10] different number of clusters. For which datasets does k-means seem to produce correct clusters for (assuming the right number of k is specified) and for which datasets does k-means fail for all values of k? 
    
    Create a pdf of the plots and return in your report. 
    """
    kmeans_dct = {}
    k_values = [2, 3, 5, 10]

    for dataset_key, (X, y) in answers['1A: datasets'].items():
        labels_dict = {}
        for k in k_values:
            labels = fit_kmeans(X, k)
            labels_dict[k] = labels
        kmeans_dct[dataset_key] = ((X, y), labels_dict)  

    
    myplt.plot_part1C(kmeans_dct, 'part1_c.jpg')

    dct = answers["1C: cluster successes"] = {"bvv": [3], "add": [3],"b":[3]} 

    dct = answers["1C: cluster failures"] = ["nc","nm"]

    """
    D. Repeat 1.C a few times and comment on which (if any) datasets seem to be sensitive to the choice of initialization for the k=2,3 cases. You do not need to add the additional plots to your report.

    Create a pdf of the plots and return in your report. 
    """

    dct = answers["1D: datasets sensitive to initialization"] = ["nc","nm"]

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)
