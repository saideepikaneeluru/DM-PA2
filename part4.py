
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
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import linkage as scipy_linkage

"""
Part 4.	
Evaluation of Hierarchical Clustering over Diverse Datasets:
In this task, you will explore hierarchical clustering over different datasets. You will also evaluate different ways to merge clusters and good ways to find the cut-off point for breaking the dendrogram.
"""

# Fill these two functions with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_hierarchical_cluster(dataset, n_clusters, linkage='ward'):
    
    data, labels = dataset
        
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
        
    hierarchical_cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    hierarchical_cluster.fit(data_scaled)
        
    return hierarchical_cluster.labels_

def fit_modified(dataset, distance_threshold, linkage_method):
    data, labels = dataset
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    hierarchical_cluster = AgglomerativeClustering(n_clusters = None, distance_threshold=distance_threshold, linkage=linkage_method)
    hierarchical_cluster.fit(data_scaled)
    
    return hierarchical_cluster.labels_

def compute():
    answers = {}

    """
    A.	Repeat parts 1.A and 1.B with hierarchical clustering. That is, write a function called fit_hierarchical_cluster (or something similar) that takes the dataset, the linkage type and the number of clusters, that trains an AgglomerativeClustering sklearn estimator and returns the label predictions. Apply the same standardization as in part 1.B. Use the default distance metric (euclidean) and the default linkage (ward).
    """
    
    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    dct = answers["4A: datasets"] = {}
    
    n_samples = 100
    seed = 42
    
    nc = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed)
    dct['nc'] = list(nc)
    nm = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
    dct['nm'] = list(nm)
    bvv = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=seed)
    dct['bvv'] = list(bvv)
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=seed)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    add = (X_aniso, y)
    dct['add'] = list(add)
    b = datasets.make_blobs(n_samples=n_samples, random_state=seed)
    dct['b'] = list(b)
    
    # dct value:  the `fit_hierarchical_cluster` function
    #dct = answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster
    dct = answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster

    """
    B.	Apply your function from 4.A and make a plot similar to 1.C with the four linkage types (single, complete, ward, centroid: rows in the figure), and use 2 clusters for all runs. Compare the results to problem 1, specifically, are there any datasets that are now correctly clustered that k-means could not handle?

    Create a pdf of the plots and return in your report. 
  
    """
    
    def fit_hierarchical_cluster_linkage(dataset, n_clusters, linkage):
    
        data, labels = dataset
            
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
            
        hierarchical_cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        hierarchical_cluster.fit(data_scaled)
            
        return hierarchical_cluster.labels_
    
    
    given_datasets = {
        "nc": nc,
        "nm": nm,
        "bvv": bvv,
        "add": add,
        "b": b
        }
        
        
    num_clusters = [2]
    dataset_keys = ['nc', 'nm', 'bvv', 'add', 'b']
    linkage_types = ['single', 'complete', 'ward', 'average']
    pdf_filename = "report_4B.pdf"
    pdf_pages = []


    fig, axes = plt.subplots(len(linkage_types), len(dataset_keys), figsize=(20, 16))
    fig.suptitle('Scatter plots for different datasets and linkage types (2 clusters)', fontsize=16)
            
    for i, linkage_type in enumerate(linkage_types):
        for j, dataset_key in enumerate(dataset_keys):
            data, labels = given_datasets[dataset_key]

            for k in num_clusters:
                predicted_labels = fit_hierarchical_cluster_linkage(given_datasets[dataset_key], n_clusters=k, linkage=linkage_type)

                ax = axes[i, j]
                ax.scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='viridis')
                ax.set_title(f'{linkage_type.capitalize()} Linkage\n{dataset_key}, k={k}')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf_pages.append(fig)
    plt.close(fig)

    with PdfPages(pdf_filename) as pdf:
        for page in pdf_pages:
            pdf.savefig(page)

    # dct value: list of dataset abbreviations (see 1.C)
    dct = answers["4B: cluster successes"] = ["nc","nm"]

    """
    C.	There are essentially two main ways to find the cut-off point for breaking the diagram: specifying the number of clusters and specifying a maximum distance. The latter is challenging to optimize for without knowing and/or directly visualizing the dendrogram, however, sometimes simple heuristics can work well. The main idea is that since the merging of big clusters usually happens when distances increase, we can assume that a large distance change between clusters means that they should stay distinct. Modify the function from part 1.A to calculate a cut-off distance before classification. Specifically, estimate the cut-off distance as the maximum rate of change of the distance between successive cluster merges (you can use the scipy.hierarchy.linkage function to calculate the linkage matrix with distances). Apply this technique to all the datasets and make a plot similar to part 4.B.
    
    Create a pdf of the plots and return in your report. 
    """
    def calculate_distance_threshold(datasets, linkage_type):
        
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(datasets)

        Z = linkage(data_scaled, method=linkage_type)
        
        merge_distances = np.diff(Z[:, 2])
        
        max_rate_change_idx = np.argmax(merge_distances)
        
        distance_threshold = Z[max_rate_change_idx, 2]
        
        return distance_threshold
    
       
    given_datasets = {
        "nc": nc,
        "nm": nm,
        "bvv": bvv,
        "add": add,
        "b": b
    }
    
    dataset_keys = ['nc', 'nm', 'bvv', 'add', 'b']
    linkage_types = ['single', 'complete', 'ward', 'average']
    pdf_filename = "report_4C.pdf"
    pdf_pages = []
    distance_thresholds = {}
    
    fig, axes = plt.subplots(len(linkage_types), len(dataset_keys), figsize=(20, 16))
    fig.suptitle('Scatter plots for different datasets and linkage types', fontsize=16)
    
    for i, linkage_type in enumerate(linkage_types):
        for j, dataset_key in enumerate(dataset_keys):
            data, labels = given_datasets[dataset_key]
            
            distance_threshold = calculate_distance_threshold(data, linkage_type)
            distance_thresholds[(dataset_key, linkage_type)] = distance_threshold
            
            predicted_labels = fit_modified(given_datasets[dataset_key], distance_threshold, linkage_type)
    
            ax = axes[i, j]
            ax.scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='viridis')
            ax.set_title(f'{linkage_type.capitalize()} Linkage\n{dataset_key}')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf_pages.append(fig)
    plt.close(fig)
    
    with PdfPages(pdf_filename) as pdf:
        for page in pdf_pages:
            pdf.savefig(page)
    
    
    # dct is the function described above in 4.C
    dct = answers["4C: modified function"] = fit_modified

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)
