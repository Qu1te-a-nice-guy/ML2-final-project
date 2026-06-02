import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_pca(scaled_data, variance_threshold=0.90):
    """Reduces data dimensionality while keeping 90% of the variance information."""
    pca = PCA(n_components=variance_threshold, random_state=42)
    pca_data = pca.fit_transform(scaled_data)
    return pca_data, pca

def get_clustering_metrics(data, k_max=8):
    """Calculates WCSS (Within-Cluster Sum of Squares) to help identify the Elbow."""
    wcss = []
    scores = []
    k_values = range(2, k_max + 1)
    for k in k_values:
        model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = model.fit_predict(data)
        wcss.append(model.inertia_)
        scores.append(silhouette_score(data, labels, sample_size=min(5000, len(data))))
    return k_values, wcss, scores

def apply_final_clustering(pca_data, original_df, k):
    """Applies KMeans with the chosen K value and appends cluster labels to the data."""
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    clusters = kmeans.fit_predict(pca_data)
    result_df = original_df.copy()
    result_df['cluster'] = clusters
    return result_df