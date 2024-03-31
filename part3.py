
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

"""
Part 3.	
Hierarchical Clustering: 
Recall from lecture that agglomerative hierarchical clustering is a greedy iterative scheme that creates clusters, i.e., distinct sets of indices of points, by gradually merging the sets based on some cluster dissimilarity (distance) measure. Since each iteration merges a set of indices there are at most n-1 mergers until the all the data points are merged into a single cluster (assuming n is the total points). This merging process of the sets of indices can be illustrated by a tree diagram called a dendrogram. Hence, agglomerative hierarchal clustering can be simply defined as a function that takes in a set of points and outputs the dendrogram.
"""

# Fill this function with code at this location. Do NOT move it.
# Change the arguments and return according to
# the question asked.


def data_index_function(data, indices_I, indices_J):
    min_distance = float('inf')  # Initialize with infinity
    
    for i in indices_I:
        for j in indices_J:
            distance = euclidean(data[i], data[j])
            if distance < min_distance:
                min_distance = distance
    
    return min_distance


def compute():
    answers = {}

    """
    A.	Load the provided dataset “hierachal_toy_data.mat” using the scipy.io.loadmat function.
    """
    toy_data = io.loadmat('hierarchical_toy_data.mat')
    

    # Assuming the data variable in mat is named 'X'
    #X = mat_data['X']  # Feature data
    #y = mat_data['y'] 
    #print(mat_data.keys())

    # return value of scipy.io.loadmat()
    answers["3A: toy data"] = toy_data

    """
    B.	Create a linkage matrix Z, and plot a dendrogram using the scipy.hierarchy.linkage and scipy.hierachy.dendrogram functions, with “single” linkage.
    """
    Z = linkage(toy_data['X'], method='single')
    #fig=plt.figure(figsize=(20,10))
    # Answer: NDArray
    answers["3B: linkage"] = Z
    #print(answers["3B: linkage"])

    # Answer: the return value of the dendogram function, dicitonary
    dendro_dict = dendrogram(Z)
    ##plt.savefig('part3)ques_A.png')
    answers["3B: dendogram"] = dendro_dict
    #print(answers["3B: dendogram"])

    """
    C.	Consider the merger of the cluster corresponding to points with index sets {I={8,2,13}} J={1,9}}. At what iteration (starting from 0) were these clusters merged? That is, what row does the merger of A correspond to in the linkage matrix Z? The rows count from 0. 
    """
    

    answers["3C: iteration"] = 4
    #print(i)
    """
    D.	Write a function that takes the data and the two index sets {I,J} above, and returns the dissimilarity given by single link clustering using the Euclidian distance metric. The function should output the same value as the 3rd column of the row found in problem 2.C.
    """
    # Answer type: a function defined above
    
    answers["3D: function"] = data_index_function
    #print(data_index_function)
    """
    E.	In the actual algorithm, deciding which clusters to merge should consider all of the available clusters at each iteration. List all the clusters as index sets, using a list of lists, 
    e.g., [{0,1,2},{3,4},{5},{6},…],  that were available when the two clusters in part 2.D were merged.
    """
    

    # List the clusters. the [{0,1,2}, {3,4}, {5}, {6}, ...] represents a list of lists.
    answers["3E: clusters"] = [{4},{6,14},{8,2,13,1,9},{5},{11},{0},{10},{3},{7},{12}]
    
    """
    F.	Single linked clustering is often criticized as producing clusters where “the rich get richer”, that is, where one cluster is continuously merging with all available points. Does your dendrogram illustrate this phenomenon?
    """


    # Answer type: string. Insert your explanation as a string.
    answers["3F: rich get richer"] = "yes"

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part3.pkl", "wb") as f:
        pickle.dump(answers, f)
