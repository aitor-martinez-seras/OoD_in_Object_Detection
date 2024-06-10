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
import hdbscan

from constants import AVAILABLE_CLUSTERING_METHODS
from custom_hyperparams import CUSTOM_HYP

# VISUALIZE = CUSTOM_HYP.clusters.VISUALIZE
# MIN_SAMPLES = CUSTOM_HYP.clusters.MIN_SAMPLES
# RANGE_OF_CLUSTERS = CUSTOM_HYP.clusters.RANGE_OF_CLUSTERS


def find_optimal_number_of_clusters_one_class_one_stride_and_return_labels(
        feature_maps: np.ndarray,
        cluster_method: str,
        metric: str,
        perf_score_metric: str,
        string_for_visualization: str,
        logger: Logger,
        visualize: Optional[bool] = False
    ) -> np.ndarray:
    assert cluster_method in AVAILABLE_CLUSTERING_METHODS, f"Invalid clustering method: {cluster_method}"
    if cluster_method == 'one':
        raise ValueError("The 'one' method is not allowed for this function")
    elif cluster_method == 'all':
        # If the method is 'all', every sample is a cluster, 
        # so return one label for each individual sample or feature map
        return np.arange(len(feature_maps))
        
    elif cluster_method == 'DBSCAN':
        cluster_class = DBSCAN
        param_to_eval = 'eps'
        density_based = True
        # if metric == 'l1':
        #     eps = np.linspace(10, 100, 100)
        # elif metric == 'l2':
        #     eps = np.linspace(0.1, 10, 100)
        # elif metric == 'cosine':
        #     eps = np.linspace(0.001, 0.1, 100)
        # else:
        #     raise ValueError('')
        a00 = np.linspace(0.0001, 0.001, 100)
        a0 = np.linspace(0.001, 0.01, 100)
        a = np.linspace(0.01, 0.1, 100)
        b = np.linspace(0.1, 1, 100)
        c = np.linspace(1, 10, 100)
        d = np.linspace(10, 100, 100)
        eps = np.concatenate((a,b,c,d))
        params = {
            'eps': eps,
            'min_samples': [CUSTOM_HYP.clusters.MIN_SAMPLES],
            'metric': [metric],
        }
        # params = optimize_dbscan(feature_maps, logger, metric)
        # return DBSCAN(**params).fit_predict(feature_maps)
    elif cluster_method.startswith('KMeans'):
        if cluster_method.split('_')[-1].isdigit():
            n_clusters = int(cluster_method.split('_')[-1])
            if n_clusters < 2:
                raise ValueError("The number of clusters must be greater than 1")
            if n_clusters > len(feature_maps):
                n_clusters = len(feature_maps)
            params = {
            'n_clusters': n_clusters,
            'random_state': 10,
            }
            return KMeans(**params).fit_predict(feature_maps)
            
        cluster_class = KMeans
        param_to_eval = 'n_clusters'
        params = {
            'n_clusters': CUSTOM_HYP.clusters.RANGE_OF_CLUSTERS,
            'random_state': [10],
        }
        # params = optimize_kmeans(feature_maps, logger)
        # return KMeans(**params).fit_predict(feature_maps)
    elif cluster_method == 'HDBSCAN':
        density_based = True
        cluster_class = HDBSCAN
        param_to_eval = 'min_cluster_size'
        params = {
            'min_cluster_size': list(range(CUSTOM_HYP.clusters.MIN_SAMPLES, 50, 1)),
            #'min_samples': [CUSTOM_HYP.clusters.MIN_SAMPLES],
            'metric': [metric],
        }
        # params = optimize_hdbscan(feature_maps, logger, metric)
        # return HDBSCAN(**params).fit_predict(feature_maps)
    elif cluster_method == 'AgglomerativeClustering':
        cluster_class = AgglomerativeClustering
        param_to_eval = 'n_clusters'
        params = {
            'n_clusters': CUSTOM_HYP.clusters.RANGE_OF_CLUSTERS,
            'metric': [metric],
            'linkage': ['complete']
            # Or
            # 'metric': ['euclidean'],
            # 'linkage': ['ward']
        }
        # params = optimize_agglomerative_clustering(feature_maps, logger, metric)
        # return AgglomerativeClustering(**params).fit_predict(feature_maps)
    elif cluster_method == 'Birch':
        cluster_class = Birch
        param_to_eval = 'threshold'
        # Params: threshold=0.5, branching_factor=50, n_clusters=3, compute_labels=True, copy=True
        params = {
            'threshold': np.linspace(0.1, 5, 100),
            'branching_factor': [50],
            'n_clusters': [None],
            'compute_labels': [True],
        }
    elif cluster_method == 'MeanShift':
        cluster_class = MeanShift
        param_to_eval = 'bandwidth'
        cluster_all = False if CUSTOM_HYP.clusters.REMOVE_ORPHANS else True
        params = {
            'bandwidth': [None],
            #'bandwidth': np.linspace(0.5, 4.25, 20),
            'cluster_all': [cluster_all],
        }

    elif cluster_method == 'GMM':
        cluster_class = GaussianMixture
        param_to_eval = 'n_components'
        params = {
            'n_components': CUSTOM_HYP.clusters.RANGE_OF_CLUSTERS,
        }
        #raise NotImplementedError("GMM is not implemented yet")
        # params = optimize_gmm(feature_maps, logger)
        # return GaussianMixture(**params).fit_predict(feature_maps)
    elif cluster_method == 'BGMM':
        cluster_class = BayesianGaussianMixture
        param_to_eval = 'n_components'
        params = {
            'n_components': CUSTOM_HYP.clusters.RANGE_OF_CLUSTERS,
        }
        #raise NotImplementedError("GMM is not implemented yet")
        # params = optimize_bgmm(feature_maps, logger)
        # return BayesianGaussianMixture(**params).fit_predict(feature_maps)
    elif cluster_method == 'OPTICS':
        raise NotImplementedError("OPTICS is not implemented yet")
        density_based = True
        cluster_class = OPTICS
        param_to_eval = 'min_samples'
        params = {
            'min_samples': [2, 5, 10, 15, 20],
        }
    elif cluster_method == 'SpectralClustering':
        raise NotImplementedError("SpectralClustering is not implemented yet")
        # params = optimize_spectral_clustering(feature_maps, logger)
        # return SpectralClustering(**params).fit_predict(feature_maps)
    else:
        raise ValueError(f"Invalid clustering method: {cluster_method}")
    
    best_params, clustering_performance_scores = search_for_best_param(
        feature_maps=feature_maps,
        cluster_class=cluster_class,
        params=params,
        param_to_eval=param_to_eval,
        perf_score_metric=perf_score_metric,
        string_for_visualization=string_for_visualization,
        metric=metric,
        logger=logger,
        visualize=visualize,
        cluster_method=cluster_method,
        density_based=density_based,
    )

    if (np.array(clustering_performance_scores) == -1).all():
        try:
            logger.warning(f"{string_for_visualization.split('/')[-1]} -> All configurations resulted in a single cluster. Assigning all samples to the same cluster.")
        except Exception as e:
            logger.warning(f"All configurations resulted in a single cluster. Assigning all samples to the same cluster.")
        cluster_labels = np.zeros(len(feature_maps))
    else:
        cluster_labels = cluster_class(**best_params).fit_predict(feature_maps)

    return cluster_labels
    

def compute_score_for_all_possible_configurations(
        feature_maps: np.ndarray,
        cluster_class: Type[BaseEstimator],
        parameters_to_search: dict[str, list],
        param_to_eval: str,
        perf_score_metric: str,
        metric: str,
        logger: Logger,
        density_based: Optional[bool] = False,
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
    total_number_of_samples = len(feature_maps)
    for param_combination in product(*values):
        params = dict(zip(keys, param_combination))
        logger.debug(f"Testing parameters: {params}")
        # Initialize and fit the clustering algorithm with the current parameters
        clustering_algorithm = cluster_class(**params)
        try:
            cluster_labels = clustering_algorithm.fit_predict(feature_maps)
            #print(set(cluster_labels))
            # Check if the clustering was successful (i.e., more than one cluster)

            ### Check if there is at least 2 clusters and no more than n_samples-1
            if total_number_of_samples-1 > len(set(cluster_labels)) > 1:

                ### Check if there are orphan samples ###
                total_number_of_orphan_samples = 0
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

                ### Check that at least each cluster has CUSTOM_HYP.clusters.MIN_SAMPLES samples, except for orphans (-1) which are not counted
                unique_labels, counts = np.unique(cluster_labels, return_counts=True)
                for label, count in zip(unique_labels, counts):
                    if count < CUSTOM_HYP.clusters.MIN_SAMPLES and label != -1:
                        raise ValueError(f"Cluster {label} has less than {CUSTOM_HYP.clusters.MIN_SAMPLES} samples.")

                ### Compute performance score and append it to the list
                if total_number_of_samples-1 > len(set(cluster_labels)) > 1:
                    if density_based and CUSTOM_HYP.clusters.REMOVE_ORPHANS:
                        score = hdbscan.validity.validity_index(feature_maps_one_run.astype(np.float64), cluster_labels, metric=metric, d=feature_maps_one_run.shape[1])
                    else:
                        if perf_score_metric == 'silhouette':
                            score = silhouette_score(feature_maps_one_run, cluster_labels, metric=metric)
                            logger.debug(f"Silhouette score: {score}")
                        elif perf_score_metric == 'calinski_harabasz':
                            score = calinski_harabasz_score(feature_maps_one_run, cluster_labels)
                            logger.debug(f"Calinski-Harabasz score: {score}")
                        else:
                            raise ValueError(f"Invalid performance score metric: {perf_score_metric}")
                    # if CUSTOM_HYP.clusters.WEIGHT_SCORE_WITH_PERCENT_ORPHANS:
                    #     score = score * (1 - (total_number_of_orphan_samples / total_number_of_samples))
                    clustering_performance_scores.append(score)
                    param_configs.append(params)
                else:
                    clustering_performance_scores.append(defalut_score)
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


def plot_scores(clustering_performance_scores: List[float], params: dict, parameter_name: str, clustering_perf_metric: str,  filename: str, density_based: bool = False):
    plt.plot(params[parameter_name], clustering_performance_scores, marker='o')
    plt.xlabel(parameter_name)
    if density_based and CUSTOM_HYP.clusters.REMOVE_ORPHANS:
        clustering_perf_metric = f"DBCV"
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
        visualize: Optional[bool] = False,
        cluster_method: Optional[str] = '',
        density_based: Optional[bool] = False,
    ) -> dict:

    # Compute silhouette scores for all possible configurations
    clustering_performance_scores, param_configs = compute_score_for_all_possible_configurations(
        feature_maps,
        cluster_class,
        params,
        param_to_eval,
        perf_score_metric,
        metric,
        logger,
        density_based,
    )

    if CUSTOM_HYP.clusters.VISUALIZE:  # or visualize:
        #plot_scores(clustering_performance_scores, params, param_to_eval, f"{perf_score_metric}_scores.png")
        plot_scores(
            clustering_performance_scores=clustering_performance_scores,
            params=params,
            parameter_name=param_to_eval,
            clustering_perf_metric=perf_score_metric,
            filename=f"{string_for_visualization}_{cluster_method}_{perf_score_metric}_scores.png",
            density_based=density_based,
        )

    # Select best parameters
    best_param_config = param_configs[np.argmax(clustering_performance_scores)]
    logger.info(f"Best parameters: {best_param_config}")
    return best_param_config, clustering_performance_scores
