import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.preprocessing import StandardScaler
import warnings

def fit_kmeans(data, n_clusters):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    predicted_labels = kmeans.fit_predict(scaled_data)
    return predicted_labels

def load_datasets():
    datasets = {}
    noisy_circles = make_circles(n_samples=100, factor=0.5, noise=0.05, random_state=42)
    datasets['noisy_circles'] = noisy_circles
    noisy_moons = make_moons(n_samples=100, noise=0.05, random_state=42)
    datasets['noisy_moons'] = noisy_moons
    blobs_varied = make_blobs(n_samples=100, cluster_std=[1.0, 2.5, 0.5], random_state=42)
    datasets['blobs_varied'] = blobs_varied
    aniso = (np.dot(blobs_varied[0], [[0.6, -0.6], [-0.4, 0.8]]), blobs_varied[1])
    datasets['aniso'] = aniso
    blobs = make_blobs(n_samples=100, random_state=42)
    datasets['blobs'] = blobs
    return datasets

def plot_scatter(data, labels, k_values):
    fig, axs = plt.subplots(len(k_values), len(data), figsize=(15, 12))
    for i, k in enumerate(k_values):
        for j, (name, (X, _)) in enumerate(data.items()):
            predicted_labels = fit_kmeans(X, k)
            ax = axs[i, j] if len(k_values) > 1 else axs[j]
            ax.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis', s=50, alpha=0.5)
            ax.set_title(f'{name}, k={k}')
    plt.tight_layout()
    plt.show()

def compute():
    answers = {}
    datasets = load_datasets()
    answers["1A: datasets"] = list(datasets.keys())
    
    answers["1B: fit_kmeans"] = fit_kmeans

    k_values = [2, 3, 5, 10]
    plot_scatter(datasets, None, k_values)

    # Evaluate which datasets produce correct clusters for specified k
    cluster_successes = {}
    cluster_failures = []
    for name, (X, _) in datasets.items():
        success = []
        for k in k_values:
            predicted_labels = fit_kmeans(X, k)
            if len(np.unique(predicted_labels)) == k:
                success.append(k)
            else:
                cluster_failures.append(name)
        if success:
            cluster_successes[name] = success
    answers["1C: cluster successes" ] = cluster_successes
    answers["1C: cluster failures"] = cluster_failures

    # Determine datasets sensitive to initialization
    datasets_sensitive_init = []
    for _ in range(3):  # Repeat a few times
        for name, (X, _) in datasets.items():
            predicted_labels1 = fit_kmeans(X, 2)
            predicted_labels2 = fit_kmeans(X, 3)
            if not np.array_equal(predicted_labels1, predicted_labels2):
                datasets_sensitive_init.append(name)
    answers["1D: datasets sensitive to initialization"] = datasets_sensitive_init

    return answers

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        answers = compute()
        print(answers)
