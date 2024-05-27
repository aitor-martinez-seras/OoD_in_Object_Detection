from typing import List, Optional, Tuple, Type
from logging import Logger
from itertools import product

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, OPTICS
from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt

from constants import AVAILABLE_CLUSTERING_METHODS

VISUALIZE = True
MIN_SAMPLES = 10
RANGE_OF_CLUSTERS = range(2, 11)

#AVAILABLE_CLUSTERING_METHODS = ['one','all','DBSCAN', 'KMeans', 'GMM', 'HDBSCAN', 'OPTICS', 'SpectralClustering', 'AgglomerativeClustering']
def find_optimal_number_of_clusters_one_class_one_stride_and_return_labels(
        feature_maps: np.ndarray,
        cluster_method: str,
        metric: str, 
        logger: Logger
    ) -> np.ndarray:
    assert cluster_method in AVAILABLE_CLUSTERING_METHODS, f"Invalid clustering method: {cluster_method}"
    if cluster_method == 'one':
        raise ValueError("The 'one' method is not allowed for this function")
    elif cluster_method == 'all':
        return ValueError("The 'all' method is not allowed for this function")
    elif cluster_method == 'DBSCAN':
        params = optimize_dbscan(feature_maps, logger, metric)
        return DBSCAN(**params).fit_predict(feature_maps)
    elif cluster_method == 'KMeans':
        params = optimize_kmeans(feature_maps, logger)
        return KMeans(**params).fit_predict(feature_maps)
    elif cluster_method == 'GMM':
        raise NotImplementedError("GMM is not implemented yet")
        params = optimize_gmm(feature_maps, logger)
        return GaussianMixture(**params).fit_predict(feature_maps).labels_
    elif cluster_method == 'HDBSCAN':
        params = optimize_hdbscan(feature_maps, logger, metric)
        return HDBSCAN(**params).fit_predict(feature_maps)
    elif cluster_method == 'OPTICS':
        params = optimize_optics(feature_maps, logger, metric)
        return OPTICS(**params).fit_predict(feature_maps)
    elif cluster_method == 'SpectralClustering':
        params = optimize_spectral_clustering(feature_maps, logger)
        return SpectralClustering(**params).fit_predict(feature_maps)
    elif cluster_method == 'AgglomerativeClustering':
        params = optimize_agglomerative_clustering(feature_maps, logger, metric)
        return AgglomerativeClustering(**params).fit_predict(feature_maps)
    else:
        raise ValueError(f"Invalid clustering method: {cluster_method}")
    

def compute_silhouette_score_for_all_possible_configurations(
        feature_maps: np.ndarray,
        cluster_class: Type[BaseEstimator],
        parameters_to_search: dict[str, list],
        logger: Logger
    ) -> Tuple[List[float], List[dict]]:

    silhouette_scores = []
    param_configs = []
    # Generate all combinations of parameters
    keys, values = zip(*parameters_to_search.items())
    for param_combination in product(*values):
        params = dict(zip(keys, param_combination))
        logger.debug(f"Testing parameters: {params}")
        # Initialize and fit the clustering algorithm with the current parameters
        clustering_algorithm = cluster_class(**params)
        try:
            cluster_labels = clustering_algorithm.fit_predict(feature_maps)
            
            # Check if the clustering was successful (i.e., more than one cluster)
            if len(set(cluster_labels)) > 1:
                score = silhouette_score(feature_maps, cluster_labels)
                silhouette_scores.append(score)
                param_configs.append(params)
                logger.debug(f"Silhouette score: {score}")
            else:
                logger.debug("Clustering resulted in a single cluster, skipping.")
        except Exception as e:
            logger.error(f"Error with parameters {params}: {e}")
    return silhouette_scores, param_configs


def plot_silhouette_scores(silhouette_scores: List[float], params: np.ndarray, parameter_name: str, filename: str):
    plt.plot(params[parameter_name], silhouette_scores, marker='o')
    plt.xlabel(parameter_name)
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis")
    plt.savefig(filename)
    plt.close()


def search_for_best_param(
        feature_maps: np.ndarray,
        cluster_class: Type[BaseEstimator],
        params: dict,
        param_to_eval: str,
        logger: Logger,
    ) -> dict:

    # Compute silhouette scores for all possible configurations
    silhouette_scores, param_configs = compute_silhouette_score_for_all_possible_configurations(feature_maps, cluster_class, params, logger)

    if VISUALIZE:
        plot_silhouette_scores(silhouette_scores, params, param_to_eval, "silhouette_scores.png")

    # Select best parameters
    best_param_config = param_configs[np.argmax(silhouette_scores)]
    logger.info(f"Best parameters: {best_param_config}")
    return best_param_config


def optimize_dbscan(feature_maps: np.ndarray, logger: Logger, metric: str) -> dict:
    params = {
        'eps': np.linspace(0.1, 10, 100),
        'min_samples': [MIN_SAMPLES],
        'metric': [metric],
    }
    best_params = search_for_best_param(feature_maps, DBSCAN, params, 'eps', logger)
    return best_params


def optimize_kmeans(feature_maps: np.ndarray, logger: Logger) -> dict:
    params = {
        'n_clusters': RANGE_OF_CLUSTERS,
        'random_state': [10],
    }
    best_params = search_for_best_param(feature_maps, KMeans, params, 'n_clusters', logger)
    return best_params


def optimize_hdbscan(feature_maps: np.ndarray, logger: Logger, metric: str) -> dict:
    params = {
        'min_cluster_size': RANGE_OF_CLUSTERS,
        'min_samples': [MIN_SAMPLES],
        'metric': [metric],
    }
    best_params = search_for_best_param(feature_maps, HDBSCAN, params, 'min_cluster_size', logger)
    return best_params


def optimize_optics(feature_maps: np.ndarray, logger: Logger, metric: str) -> dict:
    params = {
        'min_samples': [MIN_SAMPLES],
        'xi': np.linspace(0.05, 0.95, 10),
        'metric': [metric]
    }
    best_params = search_for_best_param(feature_maps, OPTICS, params, 'min_samples', logger)
    return best_params


def optimize_spectral_clustering(feature_maps: np.ndarray, logger: Logger) -> dict:
    params = {
        'n_clusters': RANGE_OF_CLUSTERS,
        'random_state': [10],
    }
    best_params = search_for_best_param(feature_maps, SpectralClustering, params, 'n_clusters', logger)
    return best_params


def optimize_agglomerative_clustering(feature_maps: np.ndarray, logger: Logger, metric: str) -> dict:
    params = {
        'n_clusters': RANGE_OF_CLUSTERS,
        'metric': [metric],
    }
    best_params = search_for_best_param(feature_maps, AgglomerativeClustering, params, 'n_clusters', logger)
    return best_params


def optimize_gmm(feature_maps: np.ndarray, logger: Logger) -> dict:
    raise NotImplementedError("GMM is not implemented yet")
    params = {
        'n_components': RANGE_OF_CLUSTERS,
    }
    best_params = search_for_best_param(feature_maps, GaussianMixture, params, 'n_components', logger)
    return best_params


# def optimize_dbscan(feature_maps: np.ndarray, logger: Logger) -> dict:
    
#     silhouette_scores = []
#     n_clusters = []

#     # Range of clusters to try
#     range_eps = np.linspace(0.1, 10, 100)
#     min_samples = 5

#     for eps in range_eps:
#         clusterer = DBSCAN(eps=eps, min_samples=min_samples)
#         cluster_labels = clusterer.fit_predict(feature_maps)

#         if len(set(cluster_labels)) > 1:
#             silhouette_avg = silhouette_score(feature_maps, cluster_labels)
#             silhouette_scores.append(silhouette_avg)
#             n_clusters.append(len(set(cluster_labels)))
#         else:
#             silhouette_scores.append(0)
#             n_clusters.append(1)

#     # Plot silhouette scores and with each marker plot the number of clusters
#     plt.plot(range_eps, silhouette_scores, marker='o')
#     for i, txt in enumerate(n_clusters):
#         plt.annotate(txt, (range_eps[i], silhouette_scores[i]))
#     plt.xlabel("EPS value")
#     plt.ylabel("Silhouette Score")
#     plt.title("Silhouette Analysis")
#     plt.savefig("silhouette_scores_dbscan.png")
#     plt.close()


# def optimize_kmeans(feature_maps: np.ndarray, logger: Logger) -> dict:

#     silhouette_scores = []

#     # Range of clusters to try
#     range_n_clusters = RANGE_OF_CLUSTERS

#     feature_maps = feature_maps.reshape(feature_maps.shape[0], -1)

#     for n_clusters in range_n_clusters:
#         clusterer = KMeans(n_clusters=n_clusters, random_state=10)
#         cluster_labels = clusterer.fit_predict(feature_maps)
        
#         # Measure and append the different metrics
#         silhouette_avg = silhouette_score(feature_maps, cluster_labels)
#         silhouette_scores.append(silhouette_avg)

#     # Plot silhouette scores
#     plt.plot(range_n_clusters, silhouette_scores, marker='o')
#     plt.xlabel("Number of clusters")
#     plt.ylabel("Silhouette Score")
#     plt.title("Silhouette Analysis")
#     plt.savefig("silhouette_scores.png")
#     plt.close()
