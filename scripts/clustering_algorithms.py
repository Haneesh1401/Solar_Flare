import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data # Features
# We don't use the target 'y' (true labels) for unsupervised clustering, 
# but we'll use it to compare the results.
y_true = iris.target 

# Assuming a typical number of clusters for Iris data (k=3)
n_clusters = 3

# --- K-Means Clustering ---
print("--- K-Means Clustering ---")
# Initialize and train the K-Means model
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
kmeans.fit(X)
kmeans_labels = kmeans.labels_
print(f"K-Means Cluster Labels (first 10): {kmeans_labels[:10]}")
# Print the cluster centroids
print(f"K-Means Centroids:\n{kmeans.cluster_centers_}\n")

# --- Mixtures of Gaussians (GMM) ---
print("--- Gaussian Mixture Model (GMM) ---")
# Initialize and train the GMM model (uses the Expectation-Maximization algorithm)
gmm = GaussianMixture(n_components=n_clusters, random_state=42)
gmm.fit(X)
gmm_labels = gmm.predict(X)
print(f"GMM Cluster Labels (first 10): {gmm_labels[:10]}")
# GMM also provides probabilities of each point belonging to each cluster (soft assignment)
probs = gmm.predict_proba(X)
print(f"GMM Probability of first sample belonging to each cluster: {probs[0].round(3)}\n")

# --- Hierarchical Clustering (Agglomerative) ---
print("--- Hierarchical Clustering ---")
# Agglomerative Clustering: bottom-up approach
hc = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
hc_labels = hc.fit_predict(X)
print(f"Hierarchical Cluster Labels (first 10): {hc_labels[:10]}")

# You can also visualize the hierarchy using a Dendrogram (requires Scipy linkage)
# 'ward' minimizes the variance of the clusters being merged
linked = linkage(X, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()