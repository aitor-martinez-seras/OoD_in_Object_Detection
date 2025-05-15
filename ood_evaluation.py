import time
import os
from pathlib import Path
from datetime import datetime
from typing import Type, Union, Literal, List, Tuple, Dict, Any
from logging import Logger
from collections import OrderedDict
from itertools import product

from tap import Tap
import numpy as np
import pandas as pd
import torch

import log
from ultralytics import YOLO
from ultralytics.data.build import InfiniteDataLoader

from ood_utils import configure_extra_output_of_the_model, OODMethod, LogitsMethod, DistanceMethod, NoMethod, MSP, Energy, ODIN, Sigmoid, \
    L1DistanceOneClusterPerStride, L2DistanceOneClusterPerStride, CosineDistanceOneClusterPerStride, \
    FusionMethod, UmapMethod, IvisMethodCosine, IvisMethodL1, IvisMethodL2, TripleFusionMethod
from data_utils import read_json, write_json, load_dataset_and_dataloader
from unknown_localization_utils import select_ftmaps_summarization_method, select_thresholding_method
from constants import ROOT, STORAGE_PATH, PRUEBAS_ROOT_PATH, RESULTS_PATH, OOD_METHOD_CHOICES, TARGETS_RELATED_OPTIONS, \
    AVAILABLE_CLUSTERING_METHODS, DISTANCE_METHODS, BENCHMARKS, COCO_OOD_NAME, COCO_MIXED_NAME, COCO_OWOD_TEST_NAME, \
    COMMON_COLUMNS, COCO_OOD_COLUMNS, COCO_MIX_COLUMNS, COCO_OWOD_COLUMNS, FINAL_COLUMNS, LOGITS_METHODS, DISTANCE_METHODS, \
    INDIVIDUAL_RESULTS_FILE_PATH, AVAILABLE_DATASETS, COCO_OWOD_COLUMNS_T1
from custom_hyperparams import CUSTOM_HYP, Hyperparams, hyperparams_to_dict


NOW = datetime.now().strftime("%Y%m%d_%H%M%S")

class SimpleArgumentParser(Tap):
    # MAIN OPTION TO SELECT
    ood_method: str  # OOD detection method to use. If it is a fusion method, it must be passed as 'fusion-method1-method2'.
    visualize_oods: bool = False  # visualize the OoD detection
    compute_metrics: bool = False  # compute the metrics
    benchmark: str = ''  # Benchmark to run
    # Visualization options
    visualize_clusters: bool = False  # visualize the clusters
    # Model options
    device: int  # Device to use for training on GPU. Indicate more than one to use multiple GPUs. Use -1 for CPU.
    model: Literal["n", "s", "m", "l", "x"]  # Which variant of the model YOLO to use
    model_path: str = ''  # Relative path to the model you want to use as a starting point. Deactivates using sizes.
    workers: int = 2  # Number of background threads used to load data.
    batch_size: int = 16  # Batch size.
    # Save options
    logdir: str = 'logs'  # Where to log test info (small).
    name: str = 'prueba'  # Name of this run. Used for monitoring and checkpointing
    # Benchmarks
    ood_datasets: List[str] = []  # Datasets to use for the benchmark. Options: 'coco_ood', 'coco_mixed', 'owod'
    # Hyperparameters for YOLO
    conf_thr_train: float = 0.15  # Confidence threshold for the In-Distribution configuration
    conf_thr_test: float = 0.15  # Confidence threshold for the detections
    # Hyperparameters for the OOD detection
    tpr_thr: float = 0.95  # TPR threshold for the OoD detection
    which_split: Literal['train', 'val', 'train_val'] = 'train'  # Split to use for the thresholds
    cluster_method: str = 'one'  # Clustering method to use for the distance methods. If passed with a "-", it will be used for the fusion methods. The cluster methods will be assigned sequentially to the distance methods.
    remove_orphans: bool = False  # Whether to remove orphans from the clusters
    cluster_optimization_metric: Literal['silhouette', 'calinski_harabasz'] = 'silhouette'  # Metric to use for the optimization of the clusters
    ind_info_creation_option: str = 'valid_preds_one_stride'  # How to create the in-distribution information for the distance methods
    enhanced_unk_localization: bool = False  # Whether to use enhanced unknown localization
    which_internal_activations: str = 'roi_aligned_ftmaps'  # Which internal activations to use for the OoD detection
    # Hyperparams for FUSION methods
    fusion_strategy: Literal["and", "or", "score", "none"] = "none"
    # For Logits methods
    use_values_before_sigmoid: bool = True  # Whether to use the values before the sigmoid
    # ODIN and Energy
    temperature_energy: int = 1
    temperature_odin: int = 1000
    # Datasets
    ind_dataset: str  # Dataset to use for training and validation
    ind_split: Literal['train', 'val', 'test'] = 'train'  # Split to use in the in-distribution dataset
    ood_split: Literal['train', 'val', 'test'] = 'val'  # Split to use in the out-of-distribution dataset
    owod_task_ind: Literal["", "t1", "t2", "t3", "t4", "all_task_test"] = ""  # OWOD task to use in the in-distribution dataset
    owod_task_ood: Literal["", "t1", "t2", "t3", "t4", "all_task_test"] = ""  # OWOD task to use in the out-of-distribution dataset
    # Options to load from disk
    load_ind_activations: bool = False  # load in-distribution scores from disk
    load_clusters: bool = False  # load clusters from disk
    load_thresholds: bool = False  # load thresholds from disk
    
    def configure(self):
        self.add_argument("-m", "--model", required=False)
        #self.add_argument('--ood_method', choices=OOD_METHOD_CHOICES, required=True, help='OOD detection method to use')
        self.add_argument(
            '--ood_datasets', 
            nargs='+', 
            choices=[COCO_OOD_NAME, COCO_MIXED_NAME, COCO_OWOD_TEST_NAME], 
            help="Datasets to use for the benchmark"
        )

    def process_args(self):
        
        # Check model path
        if self.model_path:
            print('Loading model from', self.model_path)
            print('Ignoring args --model --from_scratch')
            self.from_scratch = False
            self.model = ''
        else:
            if self.model == '':
                raise ValueError("You must pass a model size.")
        
        # Check OWOD tasks for OOD
        if self.owod_task_ood == 't1':
            # In case is T1, the columns are different
            global COCO_OWOD_COLUMNS
            COCO_OWOD_COLUMNS = COCO_OWOD_COLUMNS_T1

        # Check OOD datasets
        if len(self.ood_datasets) == 0:
            raise ValueError("You must pass ood_datasets")
        else:
            for dataset in self.ood_datasets:
                if dataset not in AVAILABLE_DATASETS:
                    raise ValueError(f"Invalid dataset {dataset}")
        
        # Check OOD methods
        ood_methods = self.ood_method.split('-')
        for method in ood_methods:
            if method == 'fusion':
                print('- Using a Fusion method -')
                assert self.fusion_strategy != "None", "You must pass a fusion strategy for fusion methods"
                assert self.load_clusters == False, "You cannot load clusters for fusion methods, the option is not correctly implemented"
                assert self.load_thresholds == False, "You cannot load thresholds for fusion methods"
            elif method not in OOD_METHOD_CHOICES:
                raise ValueError(f"You must select a valid OOD method for the fusion method -> {method}")
            else:
                print(f'- Using {method} -')

        # Check cluster method
        if self.cluster_method:
            fusion_cluster_methods = self.cluster_method.split('-')
            for cluster_method in fusion_cluster_methods:
                    if cluster_method not in AVAILABLE_CLUSTERING_METHODS:
                        raise ValueError("You must select a valid clustering method")
                    
        # Check benchmarks
        if self.benchmark:
            if self.benchmark not in BENCHMARKS.keys():
                raise ValueError("You must select a valid benchmark")

            if not self.visualize_oods and not self.compute_metrics and not self.benchmark:
                raise ValueError("You must pass either visualize_oods or compute_metrics or define a benchmark")
            
            if self.benchmark == 'cluster_methods':
                if self.ood_method not in DISTANCE_METHODS:
                    raise ValueError("You must select a distance method to run this benchmark")
                
            if self.benchmark == 'fusion_strategies':
                cluster_methods = self.cluster_method.split('-')
                assert len(cluster_methods) == 2, "You must pass two cluster methods for this benchmark," \
                    "first one will be used for Dist the Logit-Dist1 fusion and second for the Dist2 in Dist1-Dist2 fusion"
                for cluster_method in cluster_methods:
                    if cluster_method not in AVAILABLE_CLUSTERING_METHODS:
                        raise ValueError("You must select a valid clustering method")
                    
            if self.benchmark == 'unk_loc_enhancement':
                print('-- Enhanced UNK localization activated --')
                self.enhanced_unk_localization = True
                CUSTOM_HYP.unk.USE_UNK_ENHANCEMENT = True
        
        # Change Hyperparameters
        if self.visualize_clusters:
            print('-- Visualizing clusters activated --')
            CUSTOM_HYP.clusters.VISUALIZE = True

        if self.remove_orphans:
            print('-- Removing orphans activated --')
            CUSTOM_HYP.clusters.REMOVE_ORPHANS = True

        # For reports
        if self.enhanced_unk_localization:
            print('-- Enhanced UNK localization activated --')
            CUSTOM_HYP.unk.USE_UNK_ENHANCEMENT = True



def select_ood_detection_method(args: SimpleArgumentParser) -> Union[LogitsMethod, DistanceMethod, FusionMethod]:
    """
    Select the OOD method to use for the evaluation.
    """
    common_kwargs = {
        'iou_threshold_for_matching': CUSTOM_HYP.IOU_THRESHOLD,
        'min_conf_threshold_train': args.conf_thr_train,
        'min_conf_threshold_test': args.conf_thr_test,
        'use_values_before_sigmoid': args.use_values_before_sigmoid,
    }
    distance_methods_kwargs = {
        'agg_method': 'mean',
        'cluster_method': args.cluster_method,
        'cluster_optimization_metric': args.cluster_optimization_metric,
        'ind_info_creation_option': args.ind_info_creation_option,
        'which_internal_activations': args.which_internal_activations,
        'enhanced_unk_localization': args.enhanced_unk_localization,
        'saliency_map_computation_function': select_ftmaps_summarization_method(CUSTOM_HYP.unk.SUMMARIZATION_METHOD),
        'thresholds_out_of_saliency_map_function': select_thresholding_method(CUSTOM_HYP.unk.THRESHOLDING_METHOD),
    }
    distance_methods_kwargs.update(common_kwargs)

    if args.ood_method.startswith('fusion'):
        complete_name = args.ood_method
        complete_cluster_method = args.cluster_method
        if len(complete_name.split('-')) == 3:
            _, method1, method2 = complete_name.split('-')
            cluster_methods = complete_cluster_method.split('-')
            count_of_dist_methods = 0

            # Method1
            args.ood_method = method1
            if method1 in DISTANCE_METHODS:
                args.cluster_method = cluster_methods[count_of_dist_methods]
                count_of_dist_methods += 1
            ood_method1 = select_ood_detection_method(args)

            # Method2
            args.ood_method = method2
            if method2 in DISTANCE_METHODS:
                if len(cluster_methods) == 1:
                    count_of_dist_methods = 0
                args.cluster_method = cluster_methods[count_of_dist_methods]
            ood_method2 = select_ood_detection_method(args)
                
            # Maintain original names
            args.ood_method = complete_name
            args.cluster_method = complete_cluster_method
            return FusionMethod(ood_method1, ood_method2, args.fusion_strategy, fusion_method_name=complete_name, cluster_method=args.cluster_method, **common_kwargs)
        
        elif len(complete_name.split('-')) == 4:
            _, method1, method2, method3 = complete_name.split('-')
            cluster_methods = complete_cluster_method.split('-')
            if len(cluster_methods) == 1:
                cluster_method1 = cluster_methods[0]
                cluster_method2 = cluster_methods[0]
                cluster_method3 = cluster_methods[0]
            else:
                # Check which are distance methods and assign to them
                _i = 0
                if method1 in DISTANCE_METHODS:
                    cluster_method1 = cluster_methods[_i]
                    _i += 1
                if method2 in DISTANCE_METHODS:
                    cluster_method2 = cluster_methods[_i]
                    _i += 1
                if method3 in DISTANCE_METHODS:
                    cluster_method3 = cluster_methods[_i]

            args.ood_method = method1
            args.cluster_method = cluster_method1
            ood_method1 = select_ood_detection_method(args)
            args.ood_method = method2
            args.cluster_method = cluster_method2
            ood_method2 = select_ood_detection_method(args)
            args.ood_method = method3
            args.cluster_method = cluster_method3
            ood_method3 = select_ood_detection_method(args)
            # Maintain original names
            args.ood_method = complete_name
            args.cluster_method = complete_cluster_method
            return TripleFusionMethod(ood_method1, ood_method2, ood_method3, cluster_method=args.cluster_method, **common_kwargs)
            

    if args.ood_method == 'NoMethod':
        return NoMethod(per_class=True, per_stride=False, **common_kwargs)
    if args.ood_method == 'MSP':
        return MSP(per_class=True, per_stride=False, **common_kwargs)
    elif args.ood_method == 'Energy':
        return Energy(temper=args.temperature_energy, per_class=True, per_stride=False, **common_kwargs)
    elif args.ood_method == 'ODIN':
        return ODIN(temper=args.temperature_odin, per_class=True, per_stride=False, **common_kwargs)
    elif args.ood_method == 'Sigmoid':
        return Sigmoid(per_class=True, per_stride=False, **common_kwargs)
    elif args.ood_method == 'L1_cl_stride':
        return L1DistanceOneClusterPerStride(**distance_methods_kwargs)
    elif args.ood_method == 'L2_cl_stride':
        return L2DistanceOneClusterPerStride(**distance_methods_kwargs)
    elif args.ood_method == 'Cosine_cl_stride':
        return CosineDistanceOneClusterPerStride(**distance_methods_kwargs)
    elif args.ood_method == 'Umap':
        return UmapMethod(**distance_methods_kwargs)
    elif args.ood_method == 'L1Ivis':
        return IvisMethodL1(**distance_methods_kwargs)
    elif args.ood_method == 'L2Ivis':
        return IvisMethodL2(**distance_methods_kwargs)
    elif args.ood_method == 'CosineIvis':
        return IvisMethodCosine(**distance_methods_kwargs)
    else:
        raise NotImplementedError("Not implemented yet")


def define_paths_of_activations_thresholds_and_clusters(ood_method: OODMethod, model: YOLO, args: SimpleArgumentParser):
    """
    Define the paths where the activations, thresholds and clusters will be stored.
    """
    clusters_path = None  # Only for distance methods
    activations_str =   f'{ood_method.which_internal_activations}_conf{ood_method.min_conf_threshold_train}_{model.ckpt["train_args"]["name"]}_activations'
    activations_str_val =   f'{activations_str}_val'
    thresholds_str =    f'{ood_method.name}_conf{ood_method.min_conf_threshold_train}_{model.ckpt["train_args"]["name"]}_thresholds'
    if args.ood_method in DISTANCE_METHODS:
        clusters_str = f'{ood_method.name}_conf{ood_method.min_conf_threshold_train}_{model.ckpt["train_args"]["name"]}_clusters_{ood_method.cluster_method}_{ood_method.cluster_optimization_metric}'
        thresholds_str += f'_{ood_method.cluster_method}'
    if args.ood_method in TARGETS_RELATED_OPTIONS:
        activations_str += f'_{args.ind_info_creation_option}'
        activations_str_val += f'_{args.ind_info_creation_option}'
        thresholds_str += f'_{args.ind_info_creation_option}'
        if args.ood_method in DISTANCE_METHODS:
            clusters_str += f'_{args.ind_info_creation_option}'
    if args.use_values_before_sigmoid:
        activations_str += '_before_sigmoid'
        activations_str_val += '_before_sigmoid'
        thresholds_str += '_before_sigmoid'
    
    activations_path = STORAGE_PATH / f'{activations_str}.pt'
    activations_val_path = STORAGE_PATH / f'{activations_str_val}.pt'
    thresholds_path = STORAGE_PATH / f'{thresholds_str}.json'
    if args.ood_method in DISTANCE_METHODS:
        clusters_path = STORAGE_PATH / f'{clusters_str}.pt'

    return activations_path, thresholds_path, clusters_path, activations_val_path


def load_or_generate_and_save_activations(activations_path: Path, ood_method: Union[LogitsMethod, DistanceMethod, FusionMethod],
                                          ind_data_loader: InfiniteDataLoader, model: YOLO, device: str, logger: Logger) -> List[torch.Tensor]:
    if activations_path.exists():
        ind_activations = torch.load(activations_path)
        logger.info(f"In-distribution activations succesfully loaded from {activations_path}")
    else:
        # Generate in_distribution activations to generate thresholds
        logger.error(f"File {activations_path} does not exist. Generating in-distribution activations by iterating over the data...")
        ind_activations = ood_method.iterate_data_to_extract_ind_activations(ind_data_loader, model, device, logger)
        logger.info("In-distribution data processed")
        logger.info("Saving in-distribution activations...")
        torch.save(ind_activations, activations_path, pickle_protocol=5)
        logger.info(f"In-distribution activations succesfully saved in {activations_path}")
    return ind_activations


def obtain_ind_activations(ood_method: OODMethod, model: YOLO, device: str, in_loader: InfiniteDataLoader, activations_paths: Union[Path, List[Path]], logger: Logger, args: SimpleArgumentParser):
    """
    Load the in-distribution activations from disk if they exist. If not, generate them and save them.
    """
    # Parse the paths
    if isinstance(activations_paths, Path):
        activations_path = activations_paths
    elif isinstance(activations_paths, List):
        if len(activations_paths) == 2:
            activations_path1, activations_path2 = activations_paths
        elif len(activations_paths) == 3:
            activations_path1, activations_path2, activations_path3 = activations_paths
        #activations_path1, activations_path2 = activations_paths 
    else:
        raise ValueError("Invalid number of activations paths")
    
    # Load activations
    if args.load_ind_activations:
        # Load in_distribution activations from disk
        logger.info("Loading in-distribution activations...")
        if args.ood_method.startswith('fusion'):  # For fusion methods
            if len(args.ood_method.split('-')) == 3:
                configure_extra_output_of_the_model(model, ood_method.method1)
                ind_activations1 = load_or_generate_and_save_activations(activations_path1, ood_method.method1, in_loader, model, device, logger)
                configure_extra_output_of_the_model(model, ood_method.method2)
                ind_activations2 = load_or_generate_and_save_activations(activations_path2, ood_method.method2, in_loader, model, device, logger)
                ind_activations = [ind_activations1, ind_activations2]
            elif len(args.ood_method.split('-')) == 4:
                configure_extra_output_of_the_model(model, ood_method.method1)
                ind_activations1 = load_or_generate_and_save_activations(activations_path1, ood_method.method1, in_loader, model, device, logger)
                configure_extra_output_of_the_model(model, ood_method.method2)
                ind_activations2 = load_or_generate_and_save_activations(activations_path2, ood_method.method2, in_loader, model, device, logger)
                configure_extra_output_of_the_model(model, ood_method.method3)
                ind_activations3 = load_or_generate_and_save_activations(activations_path3, ood_method.method3, in_loader, model, device, logger)
                ind_activations = [ind_activations1, ind_activations2, ind_activations3]
        # For the rest of the methods
        else:
            ind_activations = load_or_generate_and_save_activations(activations_path, ood_method, in_loader, model, device, logger)

    # Generate in_distribution activations to generate thresholds
    else:
        if args.ood_method.startswith('fusion'):  # For fusion methods
            logger.info("Processing in-distribution data for BOTH fused methods...")
            ind_activations = ood_method.iterate_data_to_extract_ind_activations(in_loader, model, device, logger)
            torch.save(ind_activations[0], activations_path1, pickle_protocol=5)
            torch.save(ind_activations[1], activations_path2, pickle_protocol=5)
            logger.info("In-distribution data processed and saved")

        # Rest of the methods
        else:
            logger.info("Processing in-distribution data...")
            ind_activations = ood_method.iterate_data_to_extract_ind_activations(in_loader, model, device, logger)
            logger.info("In-distribution data processed")
            logger.info("Saving in-distribution activations...")
            torch.save(ind_activations, activations_path, pickle_protocol=5)
            logger.info(f"In-distribution activations succesfully saved in {activations_path}")

    return ind_activations


def execute_pipeline_for_in_distribution_configuration(ood_method: Union[LogitsMethod, DistanceMethod, FusionMethod, TripleFusionMethod], model: YOLO, device: str, 
                                               in_loader_train: InfiniteDataLoader, ind_dataloader_val: InfiniteDataLoader, logger: Logger, args: SimpleArgumentParser):
    """
    Execute the pipeline for the OOD evaluation. This includes the following steps:
    1. Extract activations from the in-distribution data
    2. Generate clusters (Only for distance methods)
    3. Compute scores
    4. Generate thresholds
    5. Save thresholds
    
    We can select to skip some of the steps by loading the activations, clusters or thresholds from disk.

    The generated thresholds (and clusters) are stored in the OODMethod object.
    """
    logger.info("Obtaining thresholds...")
    logger.flush()

    # ALL method must use validation split in order no to collapse all the thresholds to 0
    original_value_which_split = args.which_split
    if args.cluster_method == 'all' and ood_method.is_distance_method:
        logger.warning(f"Setting use_val_split_for_thresholds to True and use_train_and_val_for_thresholds to False as the cluster method is {args.cluster_method}")
        args.which_split = 'val'

    if args.ood_method.startswith('fusion'):
        activations_path_train, activations_path_val, thresholds_path, clusters_path = [], [], [], []
        complete_name = args.ood_method
        if len(complete_name.split('-')) == 3:
            _, method1, method2 = complete_name.split('-')
            args.ood_method = method1
            activations_path1_train, thresholds_path1, clusters_path1, activations_path1_val = define_paths_of_activations_thresholds_and_clusters(ood_method.method1, model, args)
            args.ood_method = method2
            activations_path2_train, thresholds_path2, clusters_path2, activations_path2_val = define_paths_of_activations_thresholds_and_clusters(
                ood_method.method2, model, args
            )
            activations_path_train.append(activations_path1_train)
            activations_path_train.append(activations_path2_train)
            activations_path_val.append(activations_path1_val)
            activations_path_val.append(activations_path2_val)
            thresholds_path.append(thresholds_path1)
            thresholds_path.append(thresholds_path2)
            if ood_method.method1.is_distance_method: clusters_path.append(clusters_path1)
            if ood_method.method2.is_distance_method: clusters_path.append(clusters_path2) 
            
        elif len(complete_name.split('-')) == 4:
            _, method1, method2, method3 = complete_name.split('-')
            args.ood_method = method1
            activations_path1_train, thresholds_path1, clusters_path1, activations_path1_val = define_paths_of_activations_thresholds_and_clusters(ood_method.method1, model, args)
            args.ood_method = method2
            activations_path2_train, thresholds_path2, clusters_path2, activations_path2_val = define_paths_of_activations_thresholds_and_clusters(ood_method.method2, model, args)
            args.ood_method = method3
            activations_path3_train, thresholds_path3, clusters_path3, activations_path3_val = define_paths_of_activations_thresholds_and_clusters(ood_method.method3, model, args)
            activations_path_train.append(activations_path1_train)
            activations_path_train.append(activations_path2_train)
            activations_path_train.append(activations_path3_train)
            activations_path_val.append(activations_path1_val)
            activations_path_val.append(activations_path2_val)
            activations_path_val.append(activations_path3_val)
            thresholds_path.append(thresholds_path1)
            thresholds_path.append(thresholds_path2)
            thresholds_path.append(thresholds_path3)
            if ood_method.method1.is_distance_method: clusters_path.append(clusters_path1)
            if ood_method.method2.is_distance_method: clusters_path.append(clusters_path2)
            if ood_method.method3.is_distance_method: clusters_path.append(clusters_path3)

        # Maintain original name
        args.ood_method = complete_name
        
    else:
        activations_path_train, thresholds_path, clusters_path, activations_path_val = define_paths_of_activations_thresholds_and_clusters(ood_method, model, args)

    ### Load the thresholds ###
    if args.load_thresholds:
        # Load thresholds from disk
        ood_method.thresholds = read_json(thresholds_path)
        logger.info(f"Thresholds succesfully loaded from {thresholds_path}")
        # For a distance method also the clusters are needed
        if ood_method.is_distance_method:
            # Load in_distribution clusters from disk
            ood_method.clusters = torch.load(clusters_path)
            logger.info(f"As we have a distance method, clusters have been also loaded from {clusters_path}")

    ### Compute thresholds ###
    else:
        
        ### 1. Obtain activations ###
        ind_activations_train = obtain_ind_activations(ood_method, model, device, in_loader_train, activations_path_train, logger, args)
        #if args.use_val_split_for_thresholds or args.use_train_and_val_for_thresholds:
        if args.which_split in ['train_val', 'val']:
            ind_activations_val = obtain_ind_activations(ood_method, model, device, ind_dataloader_val, activations_path_val, logger, args)
        else:
            ind_activations_val = None

        ### 2. Obtain scores ###
        # Distance methods need to have clusters representing the In-Distribution data and then compute the scores
        if ood_method.is_distance_method:
            
            ### 2.1. Distance methods need to obtain clusters for scores ###
            # Load the clusters
            if args.load_clusters:
                # Load in_distribution clusters from disk
                logger.info("Loading clusters...")
                if isinstance(clusters_path, list):  # Fusion method case
                    clusters = []
                    at_least_one_cluster_not_found = False
                    for c_path in clusters_path:
                        if c_path.exists():
                            clusters.append(torch.load(c_path))
                        else:
                            at_least_one_cluster_not_found = True
                    if at_least_one_cluster_not_found:
                        logger.error(f"File {c_path} does not exist. Generating clusters by using the activations...")
                        ood_method.clusters = ood_method.generate_clusters(ind_activations_train, logger)
                    else:
                        # Assign the clusters to the OOD method
                        ood_method.clusters = clusters

                else:  # Normal case
                    if clusters_path.exists():
                        ood_method.clusters = torch.load(clusters_path)
                    else:
                        logger.error(f"File {clusters_path} does not exist. Generating clusters by using the activations...")
                        ood_method.clusters = ood_method.generate_clusters(ind_activations_train, logger)
                        logger.info("Saving clusters...")
                        torch.save(ood_method.clusters, clusters_path, pickle_protocol=5)
                        logger.info(f"Clusters succesfully saved in {clusters_path}")
                logger.info(f"Clusters succesfully loaded from {clusters_path}")

            # Generate the clusters using the In-Distribution activations
            else:
                # Generate in_distribution clusters to generate thresholds for OOD method
                logger.info("Generating clusters...")
                ood_method.clusters = ood_method.generate_clusters(ind_activations_train, logger)
                if isinstance(clusters_path, list):  # Fusion method case
                    clusters = ood_method.clusters
                    for idx, c_path in enumerate(clusters_path):
                        logger.info("Saving clusters...")
                        torch.save(clusters[idx], c_path, pickle_protocol=5)
                else:    
                    logger.info("Saving clusters...")
                    torch.save(ood_method.clusters, clusters_path, pickle_protocol=5)

        # Select the activations to use for the scores
        logger.info("Generating in-distribution scores...")
        #if args.use_val_split_for_thresholds:
        if args.which_split == 'val':
            ind_activations = ind_activations_val
            #ind_scores = ood_method.compute_scores_from_activations(, logger)
        #elif args.use_train_and_val_for_thresholds:
        elif args.which_split == 'train_val':
            if args.ood_method.startswith('fusion'):
                if len(args.ood_method.split('-')) == 3:
                    ind_activations1 = concat_arrays_inside_list_of_lists(ind_activations_train[0], ind_activations_val[0], per_class=ood_method.method1.per_class, per_stride=ood_method.method1.per_stride)
                    ind_activations2 = concat_arrays_inside_list_of_lists(ind_activations_train[1], ind_activations_val[1], per_class=ood_method.method2.per_class, per_stride=ood_method.method2.per_stride)
                    ind_activations = [ind_activations1, ind_activations2]
                elif len(args.ood_method.split('-')) == 4:
                    ind_activations1 = concat_arrays_inside_list_of_lists(ind_activations_train[0], ind_activations_val[0], per_class=ood_method.method1.per_class, per_stride=ood_method.method1.per_stride)
                    ind_activations2 = concat_arrays_inside_list_of_lists(ind_activations_train[1], ind_activations_val[1], per_class=ood_method.method2.per_class, per_stride=ood_method.method2.per_stride)
                    ind_activations3 = concat_arrays_inside_list_of_lists(ind_activations_train[2], ind_activations_val[2], per_class=ood_method.method3.per_class, per_stride=ood_method.method3.per_stride)
                    ind_activations = [ind_activations1, ind_activations2, ind_activations3]
            else:
                ind_activations = concat_arrays_inside_list_of_lists(ind_activations_train, ind_activations_val, per_class=ood_method.per_class, per_stride=ood_method.per_stride)
        else:
            ind_activations = ind_activations_train
            
        # Compute the scores
        ind_scores = ood_method.compute_scores_from_activations(ind_activations, logger)

        # For the UNK proposals
        if hasattr(CUSTOM_HYP.unk, 'rank') and CUSTOM_HYP.unk.rank.USE_UNK_PROPOSALS_THR:
            logger.info("Generating scores to evaluate UNK proposals...")
            scores_for_unk_prop = ood_method.compute_scores_from_activations_for_unk_proposals(ind_activations, logger)
            logger.info("Saving UNK proposals...")

        ### 3. Obtain thresholds ###
        # Finally generate and save the thresholds
        logger.info("Generating thresholds...")
        ood_method.thresholds = ood_method.generate_thresholds(ind_scores, tpr=args.tpr_thr, logger=logger)
        if hasattr(CUSTOM_HYP.unk, 'rank'):
            if CUSTOM_HYP.unk.rank.USE_UNK_PROPOSALS_THR:
                logger.info("Generating scores to evaluate UNK proposals...")
                scores_for_unk_prop = ood_method.generate_unk_prop_thr(scores_for_unk_prop, tpr=args.tpr_thr)
                logger.info("Saving UNK proposals...")
        logger.info("Saving thresholds...")
        if args.ood_method.startswith('fusion'):
            if len(args.ood_method.split('-')) == 3:
                write_json(ood_method.thresholds[0], thresholds_path[0])
                write_json(ood_method.thresholds[1], thresholds_path[1])
            elif len(args.ood_method.split('-')) == 4:
                write_json(ood_method.thresholds[0], thresholds_path1)
                write_json(ood_method.thresholds[1], thresholds_path2)
                write_json(ood_method.thresholds[2], thresholds_path3)
        else:
            write_json(ood_method.thresholds, thresholds_path)

    # Restore the original value for 'all' cluster method in case it was changed
    if args.cluster_method == 'all' and ood_method.is_distance_method:
        args.which_split = original_value_which_split


def concat_arrays_inside_list_of_lists(*tuple_of_list_of_arrays: Tuple[List[List[np.ndarray]]], per_class: bool, per_stride: bool) -> List[List[np.ndarray]]:
    """
    Concatenate the arrays inside the lists.
    """
    if per_class:
        n_classes = len(tuple_of_list_of_arrays[0])
        if per_stride:
            n_strides = len(tuple_of_list_of_arrays[0][0])
            # Create the two lists of lists
            new_list_of_lists = [[[] for _ in range(n_strides)] for _ in range(n_classes)]
            for idx_cls in range(n_classes):
                for idx_stride in range(n_strides):
                    one_cls_one_stride_arrays = []
                    # For the current class and stride, take the arrays from all the lists that are not empty
                    for list_of_arrays in tuple_of_list_of_arrays:
                        if len(list_of_arrays[idx_cls][idx_stride]) > 0:
                            one_cls_one_stride_arrays.append(list_of_arrays[idx_cls][idx_stride])
                    # If there is at least one array, concatenate
                    if len(one_cls_one_stride_arrays) > 0:
                        new_list_of_lists[idx_cls][idx_stride] = np.concatenate(one_cls_one_stride_arrays)
                    else:
                        new_list_of_lists[idx_cls][idx_stride] = np.array([], dtype=np.float32)
        else:
            # Create the two lists of lists
            new_list_of_lists = [[] for _ in range(n_classes)]
            for idx_cls in range(n_classes):
                one_cls_arrays = []
                # For the current class, take the arrays from all the lists that are not empty
                for list_of_arrays in tuple_of_list_of_arrays:
                    if len(list_of_arrays[idx_cls]) > 0:
                        one_cls_arrays.append(list_of_arrays[idx_cls])
                # If there is at least one array, concatenate
                if len(one_cls_arrays) > 0:
                    new_list_of_lists[idx_cls] = torch.cat(one_cls_arrays)
                else:
                    new_list_of_lists[idx_cls] = torch.tensor([], dtype=torch.float32)

    return new_list_of_lists


def save_images_with_ood_detection(ood_method: OODMethod, model: YOLO, device: str, ood_loader: InfiniteDataLoader, logger: Logger):

    assert ood_method.thresholds is not None, "Thresholds must be generated or loaded before predicting with OoD detection"

    logger.info("Predicting with OOD detection...")
    
    ood_method.iterate_data_to_plot_with_ood_labels(model, ood_loader, device, logger, PRUEBAS_ROOT_PATH, NOW)


def run_eval(ood_method: OODMethod, model: YOLO, device: str, ood_loader: InfiniteDataLoader, known_classes:List[int], logger: Logger) -> Dict[str, float]:
    logger.info("Running test to compute metrics...")
    logger.flush()
    assert ood_method.thresholds is not None, "Thresholds must be generated or loaded before predicting with OoD detection"

    results = ood_method.iterate_data_to_compute_metrics(model, device, ood_loader, logger, known_classes)

    return results


######################################################################################

# Main program

######################################################################################

def main(args: SimpleArgumentParser):
    print('---------------------------- OOD Detection ----------------------------')
    # Setup logger
    logger = log.setup_logger(args)
    print('-----------------------------------------------------------------------')

    print('** Custom Hyperparameters:')
    logger.info(CUSTOM_HYP)
    print('************************')

    # TODO: This is for reproducibility 
    # torch.backends.cudnn.benchmark = True

    # Set device
    if args.device == -1:
        device = 'cpu'
    else:
        
        gpu_number = str(args.device)
        # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_number
        # logger.warning(f'CUDA_VISIBLE_DEVICES = {gpu_number}')
        # device = f'cuda:0'
        device = f'cuda:{gpu_number}'

    # In the case of GradNorm, the batch size must be 1 to enable the method
    if args.ood_method == 'GradNorm':
        args.batch_size = 1
        logger.warning(f'Batch size changed to {args.batch_size} as using GradNorm')

    # Load YOLO model
    if args.model_path:
        model_weights_path = ROOT / args.model_path
        logger.info(f"Loading model from {args.model_path} in {args.device}")
        model = YOLO(model_weights_path, task='detect')
    else:
        model_to_load = f'yolov8{args.model}.pt'
        logger.info(f"Loading model {model_to_load} in {args.device}")
        model = YOLO(model_to_load)
    model.to(device)

    # Information about the configuration of the In-Distribution 
    logger.info(f"IoU threshold for NMS for predictions set to {CUSTOM_HYP.IOU_THRESHOLD}")
    logger.info(f"In-Distribution confidence threshold: {args.conf_thr_train}")
    logger.info(f"In-Distribution dataset: {args.ind_dataset} - {args.ind_split}")

    # Load In-Distribution dataset
    ind_dataset, ind_dataloader = load_dataset_and_dataloader(
        dataset_name=args.ind_dataset,
        data_split=args.ind_split,
        batch_size=args.batch_size,
        workers=args.workers,
        owod_task=args.owod_task_ind
    )

    ind_val_dataset, ind_val_dataloader = load_dataset_and_dataloader(
        dataset_name=args.ind_dataset,
        data_split='val',
        batch_size=args.batch_size,
        workers=args.workers,
        owod_task=args.owod_task_ind
    )

    # TODO: Si metemos otro dataset habra que hacer esto de forma mas general
    known_classes = [x for x in range(ind_dataset.number_of_classes)]

    print('--------------------------------------')
    logger.info(f"Loading Out-of-Distribution datasets:")
    ood_dataloaders = []
    results_colums = COMMON_COLUMNS
    # Load Out-of-Distribution datasets
    voc_test_dataloader = None
    coco_ood_dataloader = None
    coco_mixed_dataloader = None
    coco_owod_test_dataloader = None
    owod_val_dataloader = None

    if COCO_OOD_NAME in args.ood_datasets:
        logger.info(f"******** {COCO_OOD_NAME} - {args.ood_split} ********")
        coco_ood_dataset, coco_ood_dataloader = load_dataset_and_dataloader(
            dataset_name=COCO_OOD_NAME,
            data_split=args.ood_split,
            batch_size=args.batch_size,
            workers=args.workers,
            #owod_task=args.owod_task_ood
        )
        ood_dataloaders.append(coco_ood_dataloader)
        results_colums += COCO_OOD_COLUMNS

    if COCO_MIXED_NAME in args.ood_datasets:
        logger.info(f"******** {COCO_MIXED_NAME} - {args.ood_split} ********")
        coco_mixed_dataset, coco_mixed_dataloader = load_dataset_and_dataloader(
            dataset_name=COCO_MIXED_NAME,
            data_split=args.ood_split,
            batch_size=args.batch_size,
            workers=args.workers,
            #owod_task=args.owod_task_ood
        )
        ood_dataloaders.append(coco_mixed_dataloader)
        results_colums += COCO_MIX_COLUMNS

    if COCO_OWOD_TEST_NAME in args.ood_datasets:
        logger.info(f"******** {COCO_OWOD_TEST_NAME} - {args.ood_split} - {args.owod_task_ood} ********")
        coco_owod_test_dataset, coco_owod_test_dataloader = load_dataset_and_dataloader(
            dataset_name=COCO_OWOD_TEST_NAME,
            data_split=args.ood_split,
            batch_size=args.batch_size,
            workers=args.workers,
            owod_task=args.owod_task_ood
        )
        ood_dataloaders.append(coco_owod_test_dataloader)
        results_colums += COCO_OWOD_COLUMNS
    print('--------------------------------------')

    results_colums += FINAL_COLUMNS

    ### Execution for the configuration defined in args ###
    if args.benchmark not in BENCHMARKS.keys():

        # Load the OOD detection method
        ood_method = select_ood_detection_method(args)

        # Modify internal attributes of the model to obtain the desired outputs in the extra_item
        configure_extra_output_of_the_model(model, ood_method)

        start_time = time.time()
        logger.info("Starting the execution of the OOD experiment...")
        
        ### OOD evaluation ###

        # Main function that executes the pipeline for the OOD evaluation (explained inside the function)
        execute_pipeline_for_in_distribution_configuration(ood_method, model, device, ind_dataloader, ind_val_dataloader, logger, args)

        if args.visualize_oods:
            for ood_dataloader in ood_dataloaders:
                # Save images with OoD detection (Green for In-Distribution, Red for Out-of-Distribution, Violet the Ground Truth)
                save_images_with_ood_detection(ood_method, model, device, ood_dataloader, logger)
            
        elif args.compute_metrics:
            
            results = {}
            
            # Fill the dictionary with the information of the method
            mean_n_clusters, std_n_clusters = get_mean_and_std_n_clusters(ood_method)
            fill_dict_with_method_info(results, args, ood_method, fusion_strat='None', mean_n_clus=mean_n_clusters, std_n_clus=std_n_clusters)

            # Run normal evaluation to compute the metrics and fill the dictionary with the results
            for ood_dataloader in ood_dataloaders:
                results_one_dataset = run_eval(ood_method, model, device, ood_dataloader, known_classes, logger)
                if coco_ood_dataloader == ood_dataloader:
                    dataset_name = COCO_OOD_NAME    
                elif coco_mixed_dataloader == ood_dataloader:
                    dataset_name = COCO_MIXED_NAME
                elif coco_owod_test_dataloader == ood_dataloader:
                    dataset_name = COCO_OWOD_TEST_NAME
                else:
                    raise ValueError("Unknown dataset")
                fill_dict_with_one_dataset_results(results, results_one_dataset, dataset_name)
            
            # Obtain the dictionary with the hyperparameters
            custom_hyperparams_dict = hyperparams_to_dict(CUSTOM_HYP)
            results.update(custom_hyperparams_dict)

            # Append the results to the xlsx file. Create it if it does not exist
            # Create a dataset string ordered inverse alphabetically
            datasets_str = '-'.join(sorted(args.ood_datasets)[::-1])
            append_results_to_xlsx_and_csv(results, INDIVIDUAL_RESULTS_FILE_PATH.with_name(f"{INDIVIDUAL_RESULTS_FILE_PATH.stem}_{datasets_str}"))
        
        else:
            raise ValueError("You must pass either visualize_oods or compute_metrics")
        
        end_time = time.time()
        logger.info("Total running time of experiment: {}".format(end_time - start_time))
        logger.info(CUSTOM_HYP)

    ### Benchmark execution ###
    else:
        logger.info(f"Running benchmark for {args.benchmark}")

        global_start_time = time.perf_counter()

        ### Benchmaks ###

        #########
        # Best methods benchmark
        #########
        if args.benchmark == 'best_methods':
            raise NotImplementedError("Not implemented yet")

        

        #########
        # Used tpr threshold benchmark
        #########
        elif args.benchmark == 'used_tpr':
            ## 1. Name results file
            if args.ood_method in DISTANCE_METHODS:
                results_file_name = f'{NOW}_{args.benchmark}_{args.ood_method}_{args.cluster_method}_conf_train{args.conf_thr_train}_conf_test{args.conf_thr_test}'
            else:
                results_file_name = f'{NOW}_{args.benchmark}_{args.ood_method}_conf_train{args.conf_thr_train}_conf_test{args.conf_thr_test}'
            USED_TPR_BENCHMARK = BENCHMARKS[args.benchmark]
            logger.info(f"Running benchmark for confidences {USED_TPR_BENCHMARK}")

            ## 2. Load common assets for the benchmark 
            logger.info(f"First configure and train the OOD detection method")
            # Load the OOD detection method
            ood_method = select_ood_detection_method(args)
            # Modify internal attributes of the model to obtain the desired outputs in the extra_item
            configure_extra_output_of_the_model(model, ood_method)

            ## 3. Run the benchmark
            final_results_df = pd.DataFrame(columns=results_colums)
            for tpr_thr_one_run in USED_TPR_BENCHMARK:
                results_one_run = {}
                print("-"*50)
                ## 3.1. Modify what is going to be benchmarked
                logger.info(f" *** Confidence threshold: {tpr_thr_one_run} ***")
                args.tpr_thr = tpr_thr_one_run
                # Create all the info for the configuration of the OOD detection method
                execute_pipeline_for_in_distribution_configuration(ood_method, model, device, ind_dataloader, ind_val_dataloader, logger, args)

                ## 3.2. Add info
                mean_n_clusters, std_n_clusters = get_mean_and_std_n_clusters(ood_method)
                fill_dict_with_method_info(results_one_run, args, ood_method, fusion_strat='None', mean_n_clus=mean_n_clusters, std_n_clus=std_n_clusters)

                ## 3.3. Run configuration for every dataset
                for dataloader in ood_dataloaders:
                    # Extract metrics
                    results_one_run_one_dataset = run_eval(ood_method, model, device, dataloader, known_classes, logger)
                    if coco_ood_dataloader == dataloader:
                        dataset_name = COCO_OOD_NAME    
                    elif coco_mixed_dataloader == dataloader:
                        dataset_name = COCO_MIXED_NAME
                    elif coco_owod_test_dataloader == dataloader:
                        dataset_name = COCO_OWOD_TEST_NAME
                    else:
                        raise ValueError("Unknown dataset")
                    fill_dict_with_one_dataset_results(results_one_run, results_one_run_one_dataset, dataset_name)
                
                ## 3.4. Collect results
                add_args_and_hyperparams_info(results_one_run, args, CUSTOM_HYP)
                final_results_df.loc[len(final_results_df)] = results_one_run
                print("-"*50, '\n')

        #########
        # Which split for In-Distribution scores threshold benchmark
        #########
        elif args.benchmark == 'which_split_for_ind_scores':
            ## 1. Name results file
            if args.ood_method in DISTANCE_METHODS:
                results_file_name = f'{NOW}_{args.benchmark}_{args.ood_method}_{args.cluster_method}_conf_train{args.conf_thr_train}_conf_test{args.conf_thr_test}'
            else:
                results_file_name = f'{NOW}_{args.benchmark}_{args.ood_method}_conf_train{args.conf_thr_train}_conf_test{args.conf_thr_test}'
            WHICH_SPLIT = BENCHMARKS[args.benchmark]
            logger.info(f"Running benchmark for options {WHICH_SPLIT}")

            ## 2. Load common assets for the benchmark 
            logger.info(f"First configure and train the OOD detection method")
            # Load the OOD detection method
            ood_method = select_ood_detection_method(args)
            # Modify internal attributes of the model to obtain the desired outputs in the extra_item
            configure_extra_output_of_the_model(model, ood_method)

            ## 3. Run the benchmark
            final_results_df = pd.DataFrame(columns=results_colums)
            for split in WHICH_SPLIT:
                results_one_run = {}
                print("-"*50)
                ## 3.1. Modify what is going to be benchmarked
                logger.info(f" *** Testing split: {split} ***")
                args.which_split = split
                # Create all the info for the configuration of the OOD detection method
                execute_pipeline_for_in_distribution_configuration(ood_method, model, device, ind_dataloader, ind_val_dataloader, logger, args)

                ## 3.2. Add info
                mean_n_clusters, std_n_clusters = get_mean_and_std_n_clusters(ood_method)
                fill_dict_with_method_info(results_one_run, args, ood_method, fusion_strat='None', mean_n_clus=mean_n_clusters, std_n_clus=std_n_clusters)

                ## 3.3. Run configuration for every dataset
                for dataloader in ood_dataloaders:
                    # Extract metrics
                    results_one_run_one_dataset = run_eval(ood_method, model, device, dataloader, known_classes, logger)
                    if coco_ood_dataloader == dataloader:
                        dataset_name = COCO_OOD_NAME    
                    elif coco_mixed_dataloader == dataloader:
                        dataset_name = COCO_MIXED_NAME
                    elif coco_owod_test_dataloader == dataloader:
                        dataset_name = COCO_OWOD_TEST_NAME
                    else:
                        raise ValueError("Unknown dataset")
                    fill_dict_with_one_dataset_results(results_one_run, results_one_run_one_dataset, dataset_name)
                
                ## 3.4. Collect results
                add_args_and_hyperparams_info(results_one_run, args, CUSTOM_HYP)
                final_results_df.loc[len(final_results_df)] = results_one_run
                print("-"*50, '\n')
        
        #########
        # Confidence threshold train benchmark
        #########
        elif args.benchmark == 'conf_thr_train':
            ## 1. Name results file
            if args.ood_method in DISTANCE_METHODS:
                results_file_name = f'{NOW}_{args.benchmark}_{args.ood_method}_{args.cluster_method}'
            else:
                results_file_name = f'{NOW}_{args.benchmark}_{args.ood_method}'
            CONF_THR_TRAIN_BENCHMARK = BENCHMARKS[args.benchmark]
            logger.info(f"Running benchmark for confidences {CONF_THR_TRAIN_BENCHMARK}")

            ## 2. Load common assets for the benchmark 
            logger.info(f"First configure and train the OOD detection method")
            # Load the OOD detection method
            ood_method = select_ood_detection_method(args)
            # Modify internal attributes of the model to obtain the desired outputs in the extra_item
            configure_extra_output_of_the_model(model, ood_method)

            ## 3. Run the benchmark
            final_results_df = pd.DataFrame(columns=results_colums)
            for conf_threshold in CONF_THR_TRAIN_BENCHMARK:
                results_one_run = {}
                print("-"*50)
                ## 3.1. Modify what is going to be benchmarked
                logger.info(f" *** Confidence threshold train: {conf_threshold} ***")
                args.conf_thr_train = conf_threshold
                ood_method.min_conf_threshold_train = conf_threshold
                # Create all the info for the configuration of the OOD detection method
                execute_pipeline_for_in_distribution_configuration(ood_method, model, device, ind_dataloader, ind_val_dataloader, logger, args)

                ## 3.2. Add info
                mean_n_clusters, std_n_clusters = get_mean_and_std_n_clusters(ood_method)
                fill_dict_with_method_info(results_one_run, args, ood_method, fusion_strat='None', mean_n_clus=mean_n_clusters, std_n_clus=std_n_clusters)

                ## 3.3. Run configuration for every dataset
                for dataloader in ood_dataloaders:
                    # Extract metrics
                    results_one_run_one_dataset = run_eval(ood_method, model, device, dataloader, known_classes, logger)
                    if coco_ood_dataloader == dataloader:
                        dataset_name = COCO_OOD_NAME    
                    elif coco_mixed_dataloader == dataloader:
                        dataset_name = COCO_MIXED_NAME
                    elif coco_owod_test_dataloader == dataloader:
                        dataset_name = COCO_OWOD_TEST_NAME
                    else:
                        raise ValueError("Unknown dataset")
                    fill_dict_with_one_dataset_results(results_one_run, results_one_run_one_dataset, dataset_name)
                
                ## 3.4. Collect results
                add_args_and_hyperparams_info(results_one_run, args, CUSTOM_HYP)
                final_results_df.loc[len(final_results_df)] = results_one_run
                print("-"*50, '\n')

        #########
        # Confidence threshold test benchmark
        #########
        elif args.benchmark == 'conf_thr_test':
            ## 1. Name results file
            if args.ood_method in DISTANCE_METHODS:
                results_file_name = f'{NOW}_{args.benchmark}_{args.ood_method}_{args.cluster_method}'
            else:
                results_file_name = f'{NOW}_{args.benchmark}_{args.ood_method}'
            CONF_THR_TEST_BENCHMARK = BENCHMARKS[args.benchmark]
            logger.info(f"Running benchmark for confidences {CONF_THR_TEST_BENCHMARK}")

            ## 2. Load common assets for the benchmark 
            logger.info(f"First configure and train the OOD detection method")
            # Load the OOD detection method
            ood_method = select_ood_detection_method(args)
            # Modify internal attributes of the model to obtain the desired outputs in the extra_item
            configure_extra_output_of_the_model(model, ood_method)
            # Create all the info for the configuration of the OOD detection method
            execute_pipeline_for_in_distribution_configuration(ood_method, model, device, ind_dataloader, ind_val_dataloader, logger, args)

            ## 3. Run the benchmark
            final_results_df = pd.DataFrame(columns=results_colums)
            for conf_threshold in CONF_THR_TEST_BENCHMARK:
                results_one_run = {}
                print("-"*50)
                ## 3.1. Modify what is going to be benchmarked
                logger.info(f" *** Confidence threshold test: {conf_threshold} ***")
                args.conf_thr_test = conf_threshold
                ood_method.min_conf_threshold_test = conf_threshold

                ## 3.2. Add info
                mean_n_clusters, std_n_clusters = get_mean_and_std_n_clusters(ood_method)
                fill_dict_with_method_info(results_one_run, args, ood_method, mean_n_clus=mean_n_clusters, std_n_clus=std_n_clusters)

                ## 3.3. Run configuration for every dataset
                for dataloader in ood_dataloaders:
                    # Extract metrics
                    results_one_run_one_dataset = run_eval(ood_method, model, device, dataloader, known_classes, logger)
                    if coco_ood_dataloader == dataloader:
                        dataset_name = COCO_OOD_NAME    
                    elif coco_mixed_dataloader == dataloader:
                        dataset_name = COCO_MIXED_NAME
                    elif coco_owod_test_dataloader == dataloader:
                        dataset_name = COCO_OWOD_TEST_NAME
                    else:
                        raise ValueError("Unknown dataset")
                    fill_dict_with_one_dataset_results(results_one_run, results_one_run_one_dataset, dataset_name)
                
                ## 3.4. Collect results
                add_args_and_hyperparams_info(results_one_run, args, CUSTOM_HYP)
                final_results_df.loc[len(final_results_df)] = results_one_run
                print("-"*50, '\n')

        #########
        # Cluster methods benchmark
        #########
        elif args.benchmark == 'cluster_methods':
            ## 1. Name results file
            results_file_name = f'{NOW}_{args.benchmark}_{args.ood_method}_{args.cluster_optimization_metric}_conf_train{args.conf_thr_train}_conf_test{args.conf_thr_test}'
            CLUSTER_METHODS_TO_TEST = BENCHMARKS[args.benchmark]
            logger.info(f"Running benchmark for methods {CLUSTER_METHODS_TO_TEST}")

            ## 2. Load common assets for the benchmark if any
            logger.info(f"First configure and train the OOD detection method")
            # Load the OOD detection method
            ood_method = select_ood_detection_method(args)
            # Modify internal attributes of the model to obtain the desired outputs in the extra_item
            configure_extra_output_of_the_model(model, ood_method)
            # Load the in-distribution activations
            activations_path_train, _, _, activations_path_val  = define_paths_of_activations_thresholds_and_clusters(ood_method, model, args)
            ind_activations_train = obtain_ind_activations(ood_method, model, device, ind_dataloader, activations_path_train, logger, args)
            ind_activations_val = obtain_ind_activations(ood_method, model, device, ind_val_dataloader, activations_path_val, logger, args)
                        
            ## 3. Run the benchmark
            final_results_df = pd.DataFrame(columns=results_colums)
            for cluster_method_one_run in CLUSTER_METHODS_TO_TEST:
                print("-"*50)
                results_one_run = {}
                ## 3.1. Modify what is going to be benchmarked
                logger.info(f"*** Cluster method: {cluster_method_one_run} ***")
                ood_method.cluster_method = cluster_method_one_run
                # Create all the info for the configuration of the OOD detection method
                if cluster_method_one_run == 'all':  # Special case
                    original_value_which_split = args.which_split
                    logger.warning("Setting use_val_split_for_thresholds to True and use_train_and_val_for_thresholds to False as the cluster method is all")
                    args.which_split = 'val'
                    # original_value_of_use_val_split = args.use_val_split_for_thresholds
                    # original_value_of_use_train_and_val = args.use_train_and_val_for_thresholds
                    # args.use_val_split_for_thresholds = True
                    # args.use_train_and_val_for_thresholds = False
                if args.which_split == 'train_val':
                    ind_activations = concat_arrays_inside_list_of_lists(ind_activations_train, ind_activations_val, per_class=ood_method.per_class, per_stride=ood_method.per_stride)
                elif args.which_split == 'val':
                    ind_activations = ind_activations_val
                elif args.which_split == 'train':
                    ind_activations = ind_activations_train
                else:
                    raise ValueError("Unknown which_split")
                ood_method.clusters = ood_method.generate_clusters(ind_activations_train, logger)
                ind_scores = ood_method.compute_scores_from_activations(ind_activations, logger)
                ood_method.thresholds = ood_method.generate_thresholds(ind_scores, tpr=args.tpr_thr, logger=logger)
                
                ## 3.2. Add info
                mean_n_clusters, std_n_clusters = get_mean_and_std_n_clusters(ood_method)
                fill_dict_with_method_info(results_one_run, args, ood_method, fusion_strat='None', mean_n_clus=mean_n_clusters, std_n_clus=std_n_clusters)

                ## 3.3. Run configuration for every dataset
                for dataloader in ood_dataloaders:
                    # Extract metrics
                    results_one_run_one_dataset = run_eval(ood_method, model, device, dataloader, known_classes, logger)
                    if coco_ood_dataloader == dataloader:
                        dataset_name = COCO_OOD_NAME    
                    elif coco_mixed_dataloader == dataloader:
                        dataset_name = COCO_MIXED_NAME
                    elif coco_owod_test_dataloader == dataloader:
                        dataset_name = COCO_OWOD_TEST_NAME
                    else:
                        raise ValueError("Unknown dataset")
                    fill_dict_with_one_dataset_results(results_one_run, results_one_run_one_dataset, dataset_name)
                
                ## 3.4. Collect results
                add_args_and_hyperparams_info(results_one_run, args, CUSTOM_HYP)
                final_results_df.loc[len(final_results_df)] = results_one_run
                print("-"*50, '\n')

                # Restore the original value for 'all' cluster method
                if cluster_method_one_run == 'all':
                    args.which_split = original_value_which_split
                    # args.use_val_split_for_thresholds = original_value_of_use_val_split
                    # args.use_train_and_val_for_thresholds = original_value_of_use_train_and_val
        
        #########
        # Logits methods benchmark
        #########
        elif args.benchmark == 'logits_methods':
            ## 1. Name results file
            results_file_name = f'{NOW}_{args.benchmark}_conf_train{args.conf_thr_train}_conf_test{args.conf_thr_test}'
            LOGITS_METHOD_TO_TEST = BENCHMARKS[args.benchmark]
            logger.info(f"Running benchmark for methods {LOGITS_METHOD_TO_TEST}")

            ## 2. Load common assets for the benchmark if any
            logger.info(f"First configure and train the OOD detection method")
            
            ## 3. Run the benchmark
            final_results_df = pd.DataFrame(columns=results_colums)
            for ood_method_name in LOGITS_METHOD_TO_TEST:
                print("-"*50)
                results_one_run = {}
                ## 3.1. Modify what is going to be benchmarked
                logger.info(f" *** Method: {ood_method_name} ***")
                args.ood_method = ood_method_name
                # Load the OOD detection method
                ood_method = select_ood_detection_method(args)
                # Modify internal attributes of the model to obtain the desired outputs in the extra_item
                configure_extra_output_of_the_model(model, ood_method)
                # Create all the info for the configuration of the OOD detection method
                execute_pipeline_for_in_distribution_configuration(ood_method, model, device, ind_dataloader,  ind_val_dataloader, logger, args)
                
                ## 3.2. Add info
                mean_n_clusters, std_n_clusters = get_mean_and_std_n_clusters(ood_method)
                fill_dict_with_method_info(results_one_run, args, ood_method, fusion_strat='None', mean_n_clus=mean_n_clusters, std_n_clus=std_n_clusters)

                ## 3.3. Run configuration for every dataset
                for dataloader in ood_dataloaders:
                    # Extract metrics
                    results_one_run_one_dataset = run_eval(ood_method, model, device, dataloader, known_classes, logger)
                    if coco_ood_dataloader == dataloader:
                        dataset_name = COCO_OOD_NAME    
                    elif coco_mixed_dataloader == dataloader:
                        dataset_name = COCO_MIXED_NAME
                    elif coco_owod_test_dataloader == dataloader:
                        dataset_name = COCO_OWOD_TEST_NAME
                    else:
                        raise ValueError("Unknown dataset")
                    fill_dict_with_one_dataset_results(results_one_run, results_one_run_one_dataset, dataset_name)
                
                ## 3.4. Collect results
                add_args_and_hyperparams_info(results_one_run, args, CUSTOM_HYP)
                final_results_df.loc[len(final_results_df)] = results_one_run
                print("-"*50, '\n')

        #########
        # Fusion strategies benchmark
        #########
        elif args.benchmark == 'fusion_strategies':
            
            ## 1. Name results file
            FUSION_STRATS_TO_TEST = list(product(*BENCHMARKS[args.benchmark]))
            names = []
            for x, _ in FUSION_STRATS_TO_TEST:
                names.extend(x.split('-'))
            names = list(OrderedDict.fromkeys(names))
            names.remove('fusion')
            names = '_'.join(names)
            results_file_name = f"{NOW}_{args.benchmark}_{names}_conf_train{args.conf_thr_train}_conf_test{args.conf_thr_test}"
            logger.info(f"Running benchmark for methods {FUSION_STRATS_TO_TEST}")

            ## 2. Load common assets for the benchmark if any
            logger.info(f"First configure and train the OOD detection method")
            # Load the OOD detection method
            args.ood_method = FUSION_STRATS_TO_TEST[0][0]
            ood_method = select_ood_detection_method(args)
            # Modify internal attributes of the model to obtain the desired outputs in the extra_item
            configure_extra_output_of_the_model(model, ood_method)
            execute_pipeline_for_in_distribution_configuration(ood_method, model, device, ind_dataloader, ind_val_dataloader, logger, args)
            
            ## 3. Run the benchmark
            final_results_df = pd.DataFrame(columns=results_colums)
            for ood_fusion_method, fusion_strat in FUSION_STRATS_TO_TEST:
                print("-"*50)
                results_one_run = {}
                ## 3.1. Modify what is going to be benchmarked
                logger.info(f" *** Fusion method and strategy: {ood_fusion_method} - {fusion_strat} ***")
                # Change to the new strat
                if ood_fusion_method != ood_method.name:  # Only load new method if it is different
                    logger.info(f"As we change method, we need to load the new method {ood_fusion_method}")
                    args.ood_method = ood_fusion_method
                    ood_method = select_ood_detection_method(args)
                    configure_extra_output_of_the_model(model, ood_method)
                    execute_pipeline_for_in_distribution_configuration(ood_method, model, device, ind_dataloader, ind_val_dataloader, logger, args)
                else:
                    logger.info(f"Method is the same, we only need to change the fusion strategy")
                # In any case, change the fusion strategy
                args.fusion_strategy = fusion_strat
                ood_method.fusion_strategy = fusion_strat
                
                ## 3.2. Add info
                if ood_method.is_distance_method:
                    if ood_method.method1.is_distance_method and ood_method.method2.is_distance_method:
                        #cluster_method_name = f"{ood_method.method1.cluster_method} + {ood_method.method2.cluster_method}"
                        mean_n_clusters1, std_n_clusters1 = get_mean_and_std_n_clusters(ood_method.method1)
                        mean_n_clusters2, std_n_clusters2 = get_mean_and_std_n_clusters(ood_method.method2)
                        mean_n_clusters = (mean_n_clusters1 + mean_n_clusters2) / 2
                        std_n_clusters = (std_n_clusters1 + std_n_clusters2) / 2
                    elif ood_method.method1.is_distance_method:
                        #cluster_method_name = ood_method.method1.cluster_method
                        mean_n_clusters, std_n_clusters = get_mean_and_std_n_clusters(ood_method.method1)
                    elif ood_method.method2.is_distance_method:
                        #cluster_method_name = ood_method.method2.cluster_method
                        mean_n_clusters, std_n_clusters = get_mean_and_std_n_clusters(ood_method.method2)
                    else:
                        raise ValueError("At least one of the methods must be a distance method if indicated in the base class")
                else:
                    #cluster_method_name = 'No Cluster Method'
                    mean_n_clusters, std_n_clusters = 0, 0

                fill_dict_with_method_info(results_one_run, args, ood_method, cluster_method=ood_method.cluster_method, mean_n_clus=mean_n_clusters, std_n_clus=std_n_clusters)

                ## 3.3. Run configuration for every dataset
                for dataloader in ood_dataloaders:
                    # Extract metrics
                    results_one_run_one_dataset = run_eval(ood_method, model, device, dataloader, known_classes, logger)
                    if coco_ood_dataloader == dataloader:
                        dataset_name = COCO_OOD_NAME    
                    elif coco_mixed_dataloader == dataloader:
                        dataset_name = COCO_MIXED_NAME
                    elif coco_owod_test_dataloader == dataloader:
                        dataset_name = COCO_OWOD_TEST_NAME
                    else:
                        raise ValueError("Unknown dataset")
                    fill_dict_with_one_dataset_results(results_one_run, results_one_run_one_dataset, dataset_name)
                
                ## 3.4. Collect results
                add_args_and_hyperparams_info(results_one_run, args, CUSTOM_HYP)
                final_results_df.loc[len(final_results_df)] = results_one_run
                print("-"*50, '\n')
        
        #########
        # Unknown localization enhancement benchmark
        #########
        elif args.benchmark == 'unk_loc_enhancement':
            ## 1. Name results file
            assert len(BENCHMARKS[args.benchmark]) == 1, "Only one combination of parameters is allowed"
            PARAMS_TO_MODIFY = BENCHMARKS[args.benchmark][0]
            results_file_name = f"{NOW}_{args.benchmark}_{args.ood_method}_conf_train{args.conf_thr_train}_conf_test{args.conf_thr_test}"
            logger.info(f"Running benchmark of unknown localization enhancement with the following combination parameters: {PARAMS_TO_MODIFY}")

            CUSTOM_HYP.BENCHMARK_MODE = True
            logger.info(f"Custom hyperparameters set to benchmark mode: {CUSTOM_HYP.BENCHMARK_MODE}")


            # Create the combinations of hyperparameters
            assert check_all_attrs_exist(CUSTOM_HYP, list(PARAMS_TO_MODIFY.keys())), "All attributes must exist in the custom hyperparameters"
            all_combinations = create_combination_dicts(PARAMS_TO_MODIFY)

            ## 2. Load common assets for the benchmark if any
            logger.info(f"First configure and train the OOD detection method")
            # Load the OOD detection method
            ood_method = select_ood_detection_method(args)
            # Modify internal attributes of the model to obtain the desired outputs in the extra_item
            configure_extra_output_of_the_model(model, ood_method)
            execute_pipeline_for_in_distribution_configuration(ood_method, model, device, ind_dataloader, ind_val_dataloader, logger, args)
            
            ## 3. Run the benchmark
            results_colums += list(PARAMS_TO_MODIFY.keys())
            final_results_df = pd.DataFrame(columns=results_colums)
            for one_combination_of_hyp in all_combinations:
                print("-"*50)
                results_one_run = {}
                ## 3.1. Modify what is going to be benchmarked
                logger.info(f" *** Running following combination: ***")
                logger.info(one_combination_of_hyp)
                modify_hyperparams_with_dict(CUSTOM_HYP, one_combination_of_hyp)

                ## 3.2. Add info
                mean_n_clusters, std_n_clusters = get_mean_and_std_n_clusters(ood_method)
                fill_dict_with_method_info(results_one_run, args, ood_method, cluster_method=ood_method.cluster_method, mean_n_clus=mean_n_clusters, std_n_clus=std_n_clusters)

                ## 3.3. Run configuration for every dataset
                for dataloader in ood_dataloaders:
                    # Extract metrics
                    results_one_run_one_dataset = run_eval(ood_method, model, device, dataloader, known_classes, logger)
                    if coco_ood_dataloader == dataloader:
                        dataset_name = COCO_OOD_NAME    
                    elif coco_mixed_dataloader == dataloader:
                        dataset_name = COCO_MIXED_NAME
                    elif coco_owod_test_dataloader == dataloader:
                        dataset_name = COCO_OWOD_TEST_NAME
                    else:
                        raise ValueError("Unknown dataset")
                    fill_dict_with_one_dataset_results(results_one_run, results_one_run_one_dataset, dataset_name)
                
                ## 3.4. Collect results
                add_args_and_hyperparams_info(results_one_run, args, CUSTOM_HYP)
                add_conmbination_of_hyperparams_to_results(results_one_run, one_combination_of_hyp)
                final_results_df.loc[len(final_results_df)] = results_one_run
                print("-"*50, '\n')

        else:
            raise ValueError(f"Unknown benchmark {args.benchmark}")
        
        # Save results
        final_results_df.to_csv(RESULTS_PATH / f'{results_file_name}.csv', index=False)
        final_results_df.to_excel(RESULTS_PATH / f'{results_file_name}.xlsx', index=False)

        # Time of execution
        global_end_time = time.perf_counter()
        logger.info("Total running time of benchmark: {}".format(global_end_time - global_start_time))


def fill_dict_with_method_info(results_one_run: Dict[str, float], args: SimpleArgumentParser, ood_method: Union[LogitsMethod, DistanceMethod, FusionMethod], **kwargs) -> None:
    results_one_run['Method'] = kwargs.get('Method', args.ood_method)
    results_one_run['which_split'] =  kwargs.get('which_split', args.which_split)
    results_one_run['conf_thr_train'] = kwargs.get('conf_thr_train', args.conf_thr_train)
    results_one_run['conf_thr_test'] = kwargs.get('conf_thr_test', args.conf_thr_test)
    results_one_run["tpr_thr"] = kwargs.get('tpr_thr', args.tpr_thr)
    results_one_run["cluster_method"] = kwargs.get('cluster_method', ood_method.cluster_method)
    results_one_run["mean_n_clus"] = kwargs.get('mean_n_clus', 0)
    results_one_run["std_n_clus"] = kwargs.get('std_n_clus', 0)
    results_one_run["fusion_strat"] = kwargs.get('fusion_strat', args.fusion_strategy)


def fill_dict_with_one_dataset_results(results_dict: Dict[str, float], results_one_dataset: Dict[str, float], dataset_name: str) -> None:
    if dataset_name == COCO_OOD_NAME:
        final_results_columns =  COCO_OOD_COLUMNS
    elif dataset_name == COCO_MIXED_NAME:
        final_results_columns = COCO_MIX_COLUMNS
    elif dataset_name == COCO_OWOD_TEST_NAME:
        final_results_columns = COCO_OWOD_COLUMNS
    else:
        raise ValueError("Unknown dataset")

    # Fill the columns
    for res_col in results_one_dataset.keys():
        for col in final_results_columns:
            if col.startswith(res_col):
                results_dict[col] = results_one_dataset[res_col]
                break


def add_args_and_hyperparams_info(results_dict: Dict[str, float], args: SimpleArgumentParser, custom_hyp: Type[Hyperparams]) -> None:
    results_dict['Model'] = args.model_path if args.model_path else f'yolov8{args.model}.pt'
    results_dict['args'] = str(args)
    results_dict['custom_hyp'] = str(custom_hyp)


def get_mean_and_std_n_clusters(ood_method: OODMethod) -> Tuple[float, float]:
    if ood_method.is_distance_method:
        if ood_method.per_class and ood_method.per_stride:
            n_clusters_per_class_per_stride = []
            for cluster_one_class in ood_method.clusters:
                for cluster_one_class_one_stride in cluster_one_class:
                    if len(cluster_one_class_one_stride) > 0:
                        n_clusters_one_class_one_stride = len(cluster_one_class_one_stride)
                        n_clusters_per_class_per_stride.append(n_clusters_one_class_one_stride)
            # Num clusters
            n_clusters_per_class_per_stride = np.array(n_clusters_per_class_per_stride)
            mean_n_clusters = np.mean(n_clusters_per_class_per_stride)
            std_n_clusters = np.std(n_clusters_per_class_per_stride)
    else:
        mean_n_clusters = 0
        std_n_clusters = 0
    return mean_n_clusters, std_n_clusters


def append_results_to_xlsx_and_csv(results: Dict[str, float], file_path: Path) -> None:

    # Ensure the directory exists
    file_path.parent.mkdir(parents=False, exist_ok=True)

    # Create a DataFrame with the new results
    new_results_df = pd.DataFrame([results])

    # Check if file exists
    if not file_path.exists():
        # Create a new DataFrame with the results
        combined_results_df = new_results_df
    else:
        # Read the existing file into a DataFrame
        existing_results_df = pd.read_excel(file_path)
        # Append the new results
        combined_results_df = pd.concat([existing_results_df, new_results_df], ignore_index=True)

    # Save the DataFrame to the Excel file
    combined_results_df.to_excel(file_path.with_suffix('.xlsx'), index=False)
    # Save the DataFrame to the CSV file
    combined_results_df.to_csv(file_path.with_suffix('.csv'), index=False)


def set_nested_attr(obj, attr, value):
    attrs = attr.split('.')
    for a in attrs[:-1]:
        obj = getattr(obj, a)
    setattr(obj, attrs[-1], value)


def get_nested_attr(obj, attr):
    attrs = attr.split('.')
    for a in attrs:
        obj = getattr(obj, a)
    return obj


def create_combination_dicts(params_to_combine: Dict[str, List]) -> List[Dict[str, Any]]:

    # Generate all combinations of the parameter values
    keys = params_to_combine.keys()
    values = params_to_combine.values()
    combinations = list(product(*values))

    # Create a list of dictionaries with all combinations
    combination_dicts = [dict(zip(keys, combination)) for combination in combinations]

    return combination_dicts


def check_all_attrs_exist(obj, attrs: List[str]):
    for attr in attrs:
        try:
            get_nested_attr(obj, attr)
        except AttributeError:
            return False
    return True


def modify_hyperparams_with_dict(hyperparams: Hyperparams, hyperparams_dict: Dict[str, Any]) -> Hyperparams:
    for key, value in hyperparams_dict.items():
        set_nested_attr(hyperparams, key, value)

    return hyperparams


def add_conmbination_of_hyperparams_to_results(results_dict: Dict[str, float], hyperparams_dict: Dict[str, Any]) -> None:
    for key, value in hyperparams_dict.items():
        results_dict[key] = value


if __name__ == "__main__":
    main(SimpleArgumentParser().parse_args())
