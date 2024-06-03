from typing import List, Optional, Tuple, Type
from logging import Logger
from itertools import product

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, OPTICS, Birch, MeanShift
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt

from constants import AVAILABLE_CLUSTERING_METHODS
from custom_hyperparams import CUSTOM_HYP

VISUALIZE = CUSTOM_HYP.clusters.VISUALIZE
MIN_SAMPLES = CUSTOM_HYP.clusters.MIN_SAMPLES
RANGE_OF_CLUSTERS = CUSTOM_HYP.clusters.RANGE_OF_CLUSTERS


def find_optimal_number_of_clusters_one_class_one_stride_and_return_labels(
        feature_maps: np.ndarray,
        cluster_method: str,
        metric: str,
        perf_score_metric: str,
        string_for_visualization: str,
        logger: Logger
    ) -> np.ndarray:
    assert cluster_method in AVAILABLE_CLUSTERING_METHODS, f"Invalid clustering method: {cluster_method}"
    if cluster_method == 'one':
        raise ValueError("The 'one' method is not allowed for this function")
    elif cluster_method == 'all':
        return ValueError("The 'all' method is not allowed for this function")
    elif cluster_method == 'DBSCAN':
        cluster_class = DBSCAN
        param_to_eval = 'eps'
        if metric == 'l1':
            eps = np.linspace(10, 100, 100)
        elif metric == 'l2':
            eps = np.linspace(0.1, 10, 100)
        elif metric == 'cosine':
            eps = np.linspace(0.001, 0.1, 100)
        else:
            raise ValueError('')
        params = {
            'eps': eps,
            'min_samples': [MIN_SAMPLES],
            'metric': [metric],
        }
        # params = optimize_dbscan(feature_maps, logger, metric)
        # return DBSCAN(**params).fit_predict(feature_maps)
    elif cluster_method == 'KMeans':
        cluster_class = KMeans
        param_to_eval = 'n_clusters'
        params = {
            'n_clusters': RANGE_OF_CLUSTERS,
            'random_state': [10],
        }
        # params = optimize_kmeans(feature_maps, logger)
        # return KMeans(**params).fit_predict(feature_maps)
    elif cluster_method == 'HDBSCAN':
        cluster_class = HDBSCAN
        param_to_eval = 'min_cluster_size'
        params = {
            'min_cluster_size': list(range(5,105, 5)),
            'min_samples': [MIN_SAMPLES],
            'metric': [metric],
        }
        # params = optimize_hdbscan(feature_maps, logger, metric)
        # return HDBSCAN(**params).fit_predict(feature_maps)
    elif cluster_method == 'AgglomerativeClustering':
        cluster_class = AgglomerativeClustering
        param_to_eval = 'n_clusters'
        params = {
            'n_clusters': RANGE_OF_CLUSTERS,
            # 'metric': [metric],
            # 'linkage': ['complete']
            # Or
            'metric': ['euclidean'],
            'linkage': ['ward']
        }
        # params = optimize_agglomerative_clustering(feature_maps, logger, metric)
        # return AgglomerativeClustering(**params).fit_predict(feature_maps)
    elif cluster_method == 'Birch':
        cluster_class = Birch
        param_to_eval = 'threshold'
        # Params: threshold=0.5, branching_factor=50, n_clusters=3, compute_labels=True, copy=True
        params = {
            'threshold': np.linspace(0.1, 10, 50),
            'branching_factor': [50],
            'n_clusters': [None],
            'compute_labels': [True],
        }
    elif cluster_method == 'MeanShift':
        cluster_class = MeanShift
        param_to_eval = 'bandwidth'
        params = {
            'bandwidth': [None, 1],
            'cluster_all': [False],
        }

    elif cluster_method == 'GMM':
        cluster_class = GaussianMixture
        param_to_eval = 'n_components'
        params = {
            'n_components': RANGE_OF_CLUSTERS,
        }
        #raise NotImplementedError("GMM is not implemented yet")
        # params = optimize_gmm(feature_maps, logger)
        # return GaussianMixture(**params).fit_predict(feature_maps)
    elif cluster_method == 'BGMM':
        cluster_class = BayesianGaussianMixture
        param_to_eval = 'n_components'
        params = {
            'n_components': RANGE_OF_CLUSTERS,
        }
        #raise NotImplementedError("GMM is not implemented yet")
        # params = optimize_bgmm(feature_maps, logger)
        # return BayesianGaussianMixture(**params).fit_predict(feature_maps)
    elif cluster_method == 'OPTICS':
        raise NotImplementedError("OPTICS is not implemented yet")
        # params = optimize_optics(feature_maps, logger, metric)
        # return OPTICS(**params).fit_predict(feature_maps)
    elif cluster_method == 'SpectralClustering':
        raise NotImplementedError("SpectralClustering is not implemented yet")
        # params = optimize_spectral_clustering(feature_maps, logger)
        # return SpectralClustering(**params).fit_predict(feature_maps)
    else:
        raise ValueError(f"Invalid clustering method: {cluster_method}")
    
    best_params = search_for_best_param(
        feature_maps=feature_maps,
        cluster_class=cluster_class,
        params=params,
        param_to_eval=param_to_eval,
        perf_score_metric=perf_score_metric,
        string_for_visualization=string_for_visualization,
        metric=metric,
        logger=logger
    )

    cluster_labels = cluster_class(**best_params).fit_predict(feature_maps)

    if VISUALIZE:
        # Plot histogram
        plt.hist(cluster_labels, bins=len(set(cluster_labels)))
        plt.xlabel('Cluster')
        plt.ylabel('Number of samples')
        plt.title('Cluster distribution')
        plt.savefig(f'{string_for_visualization}_cluster_distribution.png')
        plt.close()

    return cluster_labels
    

def compute_score_for_all_possible_configurations(
        feature_maps: np.ndarray,
        cluster_class: Type[BaseEstimator],
        parameters_to_search: dict[str, list],
        param_to_eval: str,
        perf_score_metric: str,
        metric: str,
        logger: Logger
    ) -> Tuple[List[float], List[dict]]:
    # Assert first than the only parameter that has a lenght greater than 1 is the one to be evaluated
    for param_name, param_values in parameters_to_search.items():
        if param_name != param_to_eval:
            if len(param_values) > 1:
                raise NotImplementedError(f"Only one parameter can be evaluated at a time for the moment." \
                    f"If {param_to_eval} is the parameter to evaluate, it must be the only one with more than one value.")
    assert len(parameters_to_search[param_to_eval]) > 1, f"Parameter {param_to_eval} must have more than one value to evaluate"

    if perf_score_metric == 'silhouette':
        defalut_score = -1
    elif perf_score_metric == 'calinski_harabasz':
        defalut_score = 0
    else:
        raise ValueError(f"Invalid performance score metric: {perf_score_metric}")
    clustering_performance_scores = []
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
            #print(set(cluster_labels))
            # Check if the clustering was successful (i.e., more than one cluster)

            # Check if there are orphan samples
            if len(set(cluster_labels)) > 1:
                if -1 in set(cluster_labels) and CUSTOM_HYP.clusters.REMOVE_ORPHANS:
                    logger.debug("Some samples were not assigned to any cluster (label = -1).")
                    
                    # Check if the number of orphan samples is greater than the maximum allowed
                    total_number_of_orphan_samples = np.sum(cluster_labels == -1)
                    if total_number_of_orphan_samples > CUSTOM_HYP.clusters.MAX_PERCENT_OF_ORPHANS * len(feature_maps):
                        raise ValueError(f"More than {CUSTOM_HYP.clusters.MAX_PERCENT_OF_ORPHANS*100:.0f}% of the samples were not assigned to any cluster.")
                    feature_maps_one_run = feature_maps[cluster_labels != -1]
                    cluster_labels = cluster_labels[cluster_labels != -1]

                else:  # No orphan samples
                    feature_maps_one_run = feature_maps

                # Compute performance score and append it to the list
                if perf_score_metric == 'silhouette':
                    score = silhouette_score(feature_maps_one_run, cluster_labels, metric=metric)
                    logger.debug(f"Silhouette score: {score}")
                elif perf_score_metric == 'calinski_harabasz':
                    score = calinski_harabasz_score(feature_maps_one_run, cluster_labels, metric=metric)
                    logger.debug(f"Calinski-Harabasz score: {score}")
                else:
                    raise ValueError(f"Invalid performance score metric: {perf_score_metric}")
                clustering_performance_scores.append(score)
                param_configs.append(params)
            
            # If there is only one cluster, assign the default score to the configuration
            else:
                clustering_performance_scores.append(defalut_score)
                param_configs.append(params)
                logger.debug("Clustering resulted in a single cluster, skipping.")

        except Exception as e:
            logger.error(f"Error with parameters {params}: {e}")
            clustering_performance_scores.append(defalut_score)
            param_configs.append(params)

    return clustering_performance_scores, param_configs


def plot_scores(clustering_performance_scores: List[float], params: dict, parameter_name: str, clustering_perf_metric: str,  filename: str):
    plt.plot(params[parameter_name], clustering_performance_scores, marker='o')
    plt.xlabel(parameter_name)
    plt.ylabel(f"{clustering_perf_metric} Score")
    plt.title(f"{clustering_perf_metric} Analysis")
    plt.savefig(filename)
    plt.close()


def search_for_best_param(
        feature_maps: np.ndarray,
        cluster_class: Type[BaseEstimator],
        params: dict,
        param_to_eval: str,
        perf_score_metric: str,
        string_for_visualization: str,
        metric: str,
        logger: Logger,
    ) -> dict:

    # Compute silhouette scores for all possible configurations
    clustering_performance_scores, param_configs = compute_score_for_all_possible_configurations(
        feature_maps,
        cluster_class,
        params,
        param_to_eval,
        perf_score_metric,
        metric,
        logger
    )

    if VISUALIZE:
        #plot_scores(clustering_performance_scores, params, param_to_eval, f"{perf_score_metric}_scores.png")
        plot_scores(
            clustering_performance_scores=clustering_performance_scores,
            params=params,
            parameter_name=param_to_eval,
            clustering_perf_metric=perf_score_metric,
            filename=f"{string_for_visualization}_cluster_{perf_score_metric}_scores.png"
        )

    # Select best parameters
    best_param_config = param_configs[np.argmax(clustering_performance_scores)]
    logger.info(f"Best parameters: {best_param_config}")
    return best_param_config
