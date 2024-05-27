from typing import List, Optional, Callable
from logging import Logger

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from constants import AVAILABLE_CLUSTERING_METHODS


def select_clustering_method(cluster_method: str, logger: Logger) -> Optional[Callable]:
    assert cluster_method in AVAILABLE_CLUSTERING_METHODS, f"Invalid clustering method: {cluster_method}"
    if cluster_method == 'one':
        return None
    elif cluster_method == 'all':
        return None
    elif cluster_method == 'DBSCAN':
        return DBSCAN()
    elif cluster_method == 'KMeans':
        return KMeans()

AVAILABLE_CLUSTERING_METHODS = ['one','all','DBSCAN', 'KMeans', 'GMM', 'HDBSCAN', 'OPTICS', 'SpectralClustering', 'AgglomerativeClustering']
def find_optimal_number_of_clusters_one_class_one_stride_and_return_labels(feature_maps: np.ndarray, cluster_method: Callable, logger: Logger) -> np.ndarray:
    assert cluster_method in AVAILABLE_CLUSTERING_METHODS, f"Invalid clustering method: {cluster_method}"
    if cluster_method == 'one':
        raise ValueError("The 'one' method is not allowed for this function")
    elif cluster_method == 'all':
        return ValueError("The 'all' method is not allowed for this function")
    elif cluster_method == 'DBSCAN':
        params = optimize_dbscan(feature_maps, logger)
        return DBSCAN(**params).fit_predict(feature_maps).labels_
    elif cluster_method == 'KMeans':
        params = optimize_kmeans(feature_maps, logger)
        return KMeans(**params).fit_predict(feature_maps).labels_
    

def optimize_kmeans(feature_maps: np.ndarray, logger: Logger) -> dict:
    X = np.random.rand(100, 5)

    silhouette_scores = []

    # Range of clusters to try
    range_n_clusters = range(2, 11)

    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Plot silhouette scores
    plt.plot(range_n_clusters, silhouette_scores, marker='o')
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis")
    plt.show()
