from pprint import pprint

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

# ----------------------------------------------------------------------
"""
Part 2
Comparison of Clustering Evaluation Metrics: 
In this task you will explore different methods to find a good value for k
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    distances = np.sqrt(np.sum((data - kmeans.cluster_centers_[kmeans.labels_])**2,axis=1))
    sse=np.sum(distances**2)
    return sse, kmeans.inertia_




def compute():
    # ---------------------
    answers = {}

    """
    A.	Call the make_blobs function with following parameters :(center_box=(-20,20), n_samples=20, centers=5, random_state=12).
    """
    n_samples=20
    centers=5 
    center_box=(-20, 20) 
    random_state=12
    X,label=datasets.make_blobs(n_samples=n_samples,centers=centers,center_box=center_box,random_state=random_state)
    co_1=X[0:,0:1]
    co_2=X[0:,1:]



    dct = answers["2A: blob"] = [co_1,co_2,label]
    """
    B. Modify the fit_kmeans function to return the SSE (see Equations 8.1 and 8.2 in the book).
    """
    dct = answers["2B: fit_kmeans"] = fit_kmeans
    """
    C.	Plot the SSE as a function of k for k=1,2,….,8, and choose the optimal k based on the elbow method.
    """
    
    sse_values = []
    for k in range(1, 9):
        sse, inertia = fit_kmeans(X, k)
        sse_values.append([k, sse])
        #inertia_values.append((k, inertia))

    
    dct = answers["2C: SSE plot"] = sse_values
    """
    D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k’s agree?
    """
    inertia_values = []
    for k in range(1, 9):
        _, inertia = fit_kmeans(X, k)  # Extract only the inertia value from the function's return
        inertia_values.append([k, inertia])
    dct = answers["2D: inertia plot"] =inertia_values
    dct = answers["2D: do ks agree?"] = "no"

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
