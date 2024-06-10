import time
import os
from pathlib import Path
from datetime import datetime
from typing import Type, Union, Literal, List, Tuple, Dict
from logging import Logger

from tap import Tap
import numpy as np
import torch

import log
from ultralytics import YOLO
from ultralytics.yolo.data.build import InfiniteDataLoader

from ood_utils import configure_extra_output_of_the_model, OODMethod, LogitsMethod, DistanceMethod, NoMethod, MSP, Energy, ODIN, Sigmoid, \
    L1DistanceOneClusterPerStride, L2DistanceOneClusterPerStride, GAPL2DistanceOneClusterPerStride, CosineDistanceOneClusterPerStride, \
    FusionMethod
from data_utils import read_json, write_json, load_dataset_and_dataloader
from unknown_localization_utils import select_ftmaps_summarization_method, select_thresholding_method
from constants import ROOT, STORAGE_PATH, PRUEBAS_ROOT_PATH, RESULTS_PATH, OOD_METHOD_CHOICES, TARGETS_RELATED_OPTIONS, \
    AVAILABLE_CLUSTERING_METHODS, DISTANCE_METHODS, BENCHMARKS, COCO_OOD_NAME, COCO_MIXED_NAME, COCO_OWOD_TEST_NAME, \
    COMMON_COLUMNS, COCO_OOD_COLUMNS, COCO_MIX_COLUMNS, COCO_OWOD_COLUMNS, FINAL_COLUMNS, LOGITS_METHODS, DISTANCE_METHODS
from custom_hyperparams import CUSTOM_HYP, Hyperparams


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
    benchmark_datasets: List[str] = []  # Datasets to use for the benchmark. Options: 'coco_ood', 'coco_mixed', 'owod'
    # Hyperparameters for YOLO
    conf_thr_train: float = 0.15  # Confidence threshold for the In-Distribution configuration
    conf_thr_test: float = 0.15  # Confidence threshold for the detections
    # Hyperparameters for the OOD detection
    tpr_thr: float = 0.95  # TPR threshold for the OoD detection
    which_split: Literal['train', 'val', 'train_val'] = 'train'  # Split to use for the thresholds
    # use_val_split_for_thresholds: bool = False  # Whether to use the validation split to generate the thresholds
    # use_train_and_val_for_thresholds: bool = False  # Whether to use both train and validation splits to generate the thresholds
    cluster_method: str = 'one'  # Clustering method to use for the distance methods
    remove_orphans: bool = False  # Whether to remove orphans from the clusters
    cluster_optimization_metric: Literal['silhouette', 'calinski_harabasz'] = 'silhouette'  # Metric to use for the optimization of the clusters
    ind_info_creation_option: str = 'valid_preds_one_stride'  # How to create the in-distribution information for the distance methods
    enhanced_unk_localization: bool = False  # Whether to use enhanced unknown localization
    which_internal_activations: str = 'roi_aligned_ftmaps'  # Which internal activations to use for the OoD detection
    # Hyperparams for FUSION methods
    fusion_strategy: Literal["and", "or", "score"] = "or"
    # ODIN and Energy
    temperature_energy: int = 1
    temperature_odin: int = 1000
    # Datasets
    ind_dataset: str  # Dataset to use for training and validation
    ind_split: Literal['train', 'val', 'test'] = 'train'  # Split to use in the in-distribution dataset
    ood_dataset: str  # Dataset to use for OoD detection
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
            '--benchmark_datasets', 
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
        
        # Check OOD Method
        if self.ood_method.startswith('fusion'):
            # Check 1
            _, method1, method2 = self.ood_method.split('-')
            if method1 not in OOD_METHOD_CHOICES:
                raise ValueError(f"You must select a valid OOD method for the first part of the fusion method -> {method1}")
            if method2 not in OOD_METHOD_CHOICES:
                raise ValueError(f"You must select a valid OOD method for the second part of the fusion method -> {method2}")
            # Check 2
            assert self.load_clusters == False, "You cannot load clusters for fusion methods"
            assert self.load_thresholds == False, "You cannot load thresholds for fusion methods"
        
        else:
            if self.ood_method not in OOD_METHOD_CHOICES:
                raise ValueError("You must select a valid OOD method")

        # Check cluster method
        if self.cluster_method:
            fusion_cluster_methods = self.cluster_method.split('-')
            # Case of two distance methods together
            if len(fusion_cluster_methods) == 2:
                assert self.ood_method.startswith('fusion'), "You must pass a fusion method to use two distance methods together"
                cluster_method1, cluster_method2 = fusion_cluster_methods
                if cluster_method1 not in AVAILABLE_CLUSTERING_METHODS:
                    raise ValueError("You must select a valid clustering method for the first part of the fusion method")
                if cluster_method2 not in AVAILABLE_CLUSTERING_METHODS:
                    raise ValueError("You must select a valid clustering method for the second part of the fusion method")
            # Case of only one distance method, either in fusion or alone        
            else:
                if self.cluster_method not in AVAILABLE_CLUSTERING_METHODS:
                    raise ValueError("You must select a valid clustering method")
            
        # Check usage of the split for the thresholds
        #if self.use_val_split_for_thresholds and self.use_train_and_val_for_thresholds:
        #    raise ValueError("You must select only one option for the thresholds")
        
        # Check benchmarks
        if self.benchmark:
            if self.benchmark not in BENCHMARKS.keys():
                raise ValueError("You must select a valid benchmark")

            if not self.visualize_oods and not self.compute_metrics and not self.benchmark:
                raise ValueError("You must pass either visualize_oods or compute_metrics or define a benchmark")
            
            if self.benchmark and not len(self.benchmark_datasets) > 0:
                raise ValueError("You must pass benchmark_datasets to run a benchmark")
            
            if self.benchmark == 'cluster_methods':
                if self.ood_method not in DISTANCE_METHODS:
                    raise ValueError("You must select a distance method to run this benchmark")
        
        # Change Hyperparameters
        if self.visualize_clusters:
            print('-- Visualizing clusters activated --')
            CUSTOM_HYP.clusters.VISUALIZE = True

        if self.remove_orphans:
            print('-- Removing orphans activated --')
            CUSTOM_HYP.clusters.REMOVE_ORPHANS = True



def select_ood_detection_method(args: SimpleArgumentParser) -> Union[LogitsMethod, DistanceMethod, FusionMethod]:
    """
    Select the OOD method to use for the evaluation.
    """
    common_kwargs = {
        'iou_threshold_for_matching': CUSTOM_HYP.IOU_THRESHOLD,
        'min_conf_threshold_train': args.conf_thr_train,
        'min_conf_threshold_test': args.conf_thr_test
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
        _, method1, method2 = complete_name.split('-')
        cluster_methods = complete_cluster_method.split('-')
        if len(cluster_methods) == 2:
            cluster_method1, cluster_method2 = cluster_methods
        else:  # Assign the same cluster method to both methods
            cluster_method1 = cluster_methods[0]
            cluster_method2 = cluster_methods[0]
        args.ood_method = method1
        args.cluster_method = cluster_method1
        ood_method1 = select_ood_detection_method(args)
        args.ood_method = method2
        args.cluster_method = cluster_method2
        ood_method2 = select_ood_detection_method(args)
        # Maintain original names
        args.ood_method = complete_name
        args.cluster_method = complete_cluster_method
        return FusionMethod(ood_method1, ood_method2, args.fusion_strategy, **common_kwargs)

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
    elif args.ood_method == 'GAP_L2_cl_stride':
        return GAPL2DistanceOneClusterPerStride(**distance_methods_kwargs)
    elif args.ood_method == 'Cosine_cl_stride':
        return CosineDistanceOneClusterPerStride(**distance_methods_kwargs)
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
        activations_path1, activations_path2 = activations_paths 
    else:
        raise ValueError("Invalid number of activations paths")
    
    # Load activations
    if args.load_ind_activations:
        # Load in_distribution activations from disk
        logger.info("Loading in-distribution activations...")
        if args.ood_method.startswith('fusion'):  # For fusion methods
            configure_extra_output_of_the_model(model, ood_method.method1)
            ind_activations1 = load_or_generate_and_save_activations(activations_path1, ood_method.method1, in_loader, model, device, logger)
            configure_extra_output_of_the_model(model, ood_method.method2)
            ind_activations2 = load_or_generate_and_save_activations(activations_path2, ood_method.method2, in_loader, model, device, logger)
            ind_activations = [ind_activations1, ind_activations2]
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


def execute_pipeline_for_in_distribution_configuration(ood_method: Union[LogitsMethod, DistanceMethod, FusionMethod], model: YOLO, device: str, 
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
    # original_value_of_use_val_split = args.use_val_split_for_thresholds
    # original_value_of_use_train_and_val = args.use_train_and_val_for_thresholds
    original_value_which_split = args.which_split
    if args.cluster_method == 'all' and ood_method.is_distance_method:
        logger.warning(f"Setting use_val_split_for_thresholds to True and use_train_and_val_for_thresholds to False as the cluster method is {args.cluster_method}")
        # args.use_val_split_for_thresholds = True
        # args.use_train_and_val_for_thresholds = False
        args.which_split = 'val'

    if args.ood_method.startswith('fusion'):
        activations_path_train, activations_path_val, thresholds_path = [], [], []
        complete_name = args.ood_method
        _, method1, method2 = complete_name.split('-')
        args.ood_method = method1
        activations_path1_train, thresholds_path1, _, activations_path1_val = define_paths_of_activations_thresholds_and_clusters(ood_method.method1, model, args)
        args.ood_method = method2
        activations_path2_train, thresholds_path2, clusters_path, activations_path2_val = define_paths_of_activations_thresholds_and_clusters(
            ood_method.method2, model, args
        )
        activations_path_train.append(activations_path1_train)
        activations_path_train.append(activations_path2_train)
        activations_path_val.append(activations_path1_val)
        activations_path_val.append(activations_path2_val)
        thresholds_path.append(thresholds_path1)
        thresholds_path.append(thresholds_path2)
        # Maintain original name
        args.ood_method = complete_name
        
    else:
        activations_path_train, thresholds_path, clusters_path, activations_path_val = define_paths_of_activations_thresholds_and_clusters(ood_method, model, args)

    # activations_str =   f'{ood_method.which_internal_activations}_conf{ood_method.min_conf_threshold_train}_{model.ckpt["train_args"]["name"]}_activations'
    # thresholds_str =    f'{ood_method.name}_conf{ood_method.min_conf_threshold_train}_{model.ckpt["train_args"]["name"]}_thresholds'
    # if args.ood_method in DISTANCE_METHODS:
    #     clusters_str = f'{ood_method.name}_conf{ood_method.min_conf_threshold_train}_{model.ckpt["train_args"]["name"]}_clusters_{ood_method.cluster_method}_{ood_method.cluster_optimization_metric}'
    #     thresholds_str += f'_{ood_method.cluster_method}'
    # if args.ood_method in TARGETS_RELATED_OPTIONS:
    #     activations_str += f'_{args.ind_info_creation_option}'
    #     thresholds_str += f'_{args.ind_info_creation_option}'
    #     if args.ood_method in DISTANCE_METHODS:
    #         clusters_str += f'_{args.ind_info_creation_option}'
    
    # activations_path = STORAGE_PATH / f'{activations_str}.pt'
    # thresholds_path = STORAGE_PATH / f'{thresholds_str}.json'
    # if args.ood_method in DISTANCE_METHODS:
    #     clusters_path = STORAGE_PATH / f'{clusters_str}.pt'

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
        
        # # Load activations for the thresholds
        # if args.load_ind_activations:
        #     # Load in_distribution activations from disk
        #     logger.info("Loading in-distribution activations...")
        #     if args.ood_method.startswith('fusion'):  # For fusion methods
        #         configure_extra_output_of_the_model(model, ood_method.method1)
        #         ind_activations1 = load_or_generate_and_save_activations(activations_path1, ood_method.method1, in_loader, model, device, logger)
        #         configure_extra_output_of_the_model(model, ood_method.method2)
        #         ind_activations2 = load_or_generate_and_save_activations(activations_path2, ood_method.method2, in_loader, model, device, logger)
        #         ind_activations = [ind_activations1, ind_activations2]
        #     # For the rest of the methods
        #     else:
                
        #         ind_activations = load_or_generate_and_save_activations(activations_path, ood_method, in_loader, model, device, logger)

        #     # if activations_path.exists():
        #     #     ind_activations = torch.load(activations_path)
        #     #     logger.info(f"In-distribution activations succesfully loaded from {activations_path}")
        #     # else:
        #     #     # Generate in_distribution activations to generate thresholds
        #     #     logger.error(f"File {activations_path} does not exist. Generating in-distribution activations by iterating over the data...")
        #     #     ind_activations = ood_method.iterate_data_to_extract_ind_activations(in_loader, model, device, logger)
        #     #     logger.info("In-distribution data processed")
        #     #     logger.info("Saving in-distribution activations...")
        #     #     torch.save(ind_activations, activations_path, pickle_protocol=5)
        #     #     logger.info(f"In-distribution activations succesfully saved in {activations_path}")

        # # Generate in_distribution activations to generate thresholds
        # else:
        #     if args.ood_method.startswith('fusion'):  # For fusion methods
        #         logger.info("Processing in-distribution data for BOTH fused methods...")
        #         ind_activations = ood_method.iterate_data_to_extract_ind_activations(in_loader, model, device, logger)
        #         torch.save(ind_activations[0], activations_path1, pickle_protocol=5)
        #         torch.save(ind_activations[1], activations_path2, pickle_protocol=5)

        #         # configure_extra_output_of_the_model(model, ood_method.ood_method1)
        #         # ind_activations1 = ood_method.method1.iterate_data_to_extract_ind_activations(in_loader, model, device, logger)
        #         # torch.save(ind_activations1, activations_path1, pickle_protocol=5)
        #         # configure_extra_output_of_the_model(model, ood_method.ood_method2)
        #         # ind_activations2 = ood_method.method2.iterate_data_to_extract_ind_activations(in_loader, model, device, logger)
        #         # torch.save(ind_activations2, activations_path2, pickle_protocol=5)                
        #         # ind_activations = [ind_activations1, ind_activations2]

        #         logger.info("In-distribution data processed and saved")

        #     # Rest of the methods
        #     else:
        #         logger.info("Processing in-distribution data...")
        #         ind_activations = ood_method.iterate_data_to_extract_ind_activations(in_loader, model, device, logger)
        #         logger.info("In-distribution data processed")
        #         logger.info("Saving in-distribution activations...")
        #         torch.save(ind_activations, activations_path, pickle_protocol=5)
        #         logger.info(f"In-distribution activations succesfully saved in {activations_path}")

        ### 2. Obtain scores ###
        # Distance methods need to have clusters representing the In-Distribution data and then compute the scores
        if ood_method.is_distance_method:
            
            ### 2.1. Distance methods need to obtain clusters for scores ###
            # Load the clusters
            if args.load_clusters:
                # Load in_distribution clusters from disk
                logger.info("Loading clusters...")
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
                ind_activations1 = concat_arrays_inside_list_of_lists(ind_activations_train[0], ind_activations_val[0], per_class=ood_method.method1.per_class, per_stride=ood_method.method1.per_stride)
                ind_activations2 = concat_arrays_inside_list_of_lists(ind_activations_train[1], ind_activations_val[1], per_class=ood_method.method2.per_class, per_stride=ood_method.method2.per_stride)
                ind_activations = [ind_activations1, ind_activations2]
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
        
        # # For the rest of the methods activations are the scores themselves
        # else:
        #     if args.use_val_split_for_thresholds:
        #         ind_scores = ood_method.compute_scores_from_activations(ind_activations_val, logger)
        #     elif args.use_train_and_val_for_thresholds:
        #         ind_scores = ood_method.compute_scores_from_activations(ind_activations_train + ind_activations_val, logger)
        #     else:
        #         ind_scores = ood_method.compute_scores_from_activations(ind_activations_train, logger)

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
            write_json(ood_method.thresholds[0], thresholds_path1)
            write_json(ood_method.thresholds[1], thresholds_path2)
        else:
            write_json(ood_method.thresholds, thresholds_path)

    # Restore the original value for 'all' cluster method in case it was changed
    if args.cluster_method == 'all' and ood_method.is_distance_method:
        # args.use_val_split_for_thresholds = original_value_of_use_val_split
        # args.use_train_and_val_for_thresholds = original_value_of_use_train_and_val
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

    ### Execution for the configuration defined in args ###
    if args.benchmark not in BENCHMARKS.keys():

        # Load Out-of-Distribution dataset
        ood_dataset, ood_dataloader = load_dataset_and_dataloader(
            dataset_name=args.ood_dataset,
            data_split=args.ood_split,
            batch_size=args.batch_size,
            workers=args.workers,
            owod_task=args.owod_task_ood
        )

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
            # Save images with OoD detection (Green for In-Distribution, Red for Out-of-Distribution, Violet the Ground Truth)
            save_images_with_ood_detection(ood_method, model, device, ood_dataloader, logger)
            
        elif args.compute_metrics:
            
            # Run the normal evaluation to compute the metrics
            _ = run_eval(ood_method, model, device, ood_dataloader, known_classes, logger)
        
        else:
            raise ValueError("You must pass either visualize_oods or compute_metrics")
        
        end_time = time.time()
        logger.info("Total running time of experiment: {}".format(end_time - start_time))
        logger.info(CUSTOM_HYP)

    ### Benchmark execution ###
    else:
        import pandas as pd
        logger.info(f"Running benchmark for {args.benchmark}")

        print('--------------------------------------')
        logger.info(f"Loading Out-of-Distribution datasets:")
        ood_dataloaders = []
        results_colums = COMMON_COLUMNS
        # Load Out-of-Distribution datasets
        if COCO_OOD_NAME in args.benchmark_datasets:
            logger.info(f"{COCO_OOD_NAME} - {args.ood_split}")
            coco_ood_dataset, coco_ood_dataloader = load_dataset_and_dataloader(
                dataset_name=COCO_OOD_NAME,
                data_split=args.ood_split,
                batch_size=args.batch_size,
                workers=args.workers,
                owod_task=args.owod_task_ood
            )
            ood_dataloaders.append(coco_ood_dataloader)
            results_colums += COCO_OOD_COLUMNS

        if COCO_MIXED_NAME in args.benchmark_datasets:
            logger.info(f"{COCO_MIXED_NAME} - {args.ood_split}")
            coco_mixed_dataset, coco_mixed_dataloader = load_dataset_and_dataloader(
                dataset_name=COCO_MIXED_NAME,
                data_split=args.ood_split,
                batch_size=args.batch_size,
                workers=args.workers,
                owod_task=args.owod_task_ood
            )
            ood_dataloaders.append(coco_mixed_dataloader)
            results_colums += COCO_MIX_COLUMNS

        if COCO_OWOD_TEST_NAME in args.benchmark_datasets:
            logger.info(f"{COCO_OWOD_TEST_NAME} - {args.ood_split}")
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
                # results_one_run['Method'] = args.ood_method
                # results_one_run['which_split'] = args.which_split
                # results_one_run['conf_thr_train'] = args.conf_thr_train
                # results_one_run['conf_thr_test'] = args.conf_thr_test
                # results_one_run["tpr_thr"] = args.tpr_thr
                # results_one_run["cluster_method"] = ood_method.cluster_method
                # results_one_run["mean_n_clus"] = mean_n_clusters
                # results_one_run["std_n_clus"] = std_n_clusters
                # results_one_run["fusion_strat"] = 'None'

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
                    ind_activations = concat_arrays_inside_list_of_lists(ind_activations_train, ind_activations_val)
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
            from itertools import product
            from collections import OrderedDict
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
                    args.ood_method = ood_fusion_method
                    ood_method = select_ood_detection_method(args)
                    configure_extra_output_of_the_model(model, ood_method)
                    execute_pipeline_for_in_distribution_configuration(ood_method, model, device, ind_dataloader, ind_val_dataloader, logger, args)
                # In any case, change the fusion strategy
                args.fusion_strategy = fusion_strat
                ood_method.fusion_strategy = fusion_strat
                
                ## 3.2. Add info
                if ood_method.is_distance_method:
                    if ood_method.method1.is_distance_method and ood_method.method2.is_distance_method:
                        cluster_method_name = f"{ood_method.method1.cluster_method} + {ood_method.method2.cluster_method}"
                        mean_n_clusters1, std_n_clusters1 = get_mean_and_std_n_clusters(ood_method.method1)
                        mean_n_clusters2, std_n_clusters2 = get_mean_and_std_n_clusters(ood_method.method2)
                        mean_n_clusters = (mean_n_clusters1 + mean_n_clusters2) / 2
                        std_n_clusters = (std_n_clusters1 + std_n_clusters2) / 2
                    elif ood_method.method1.is_distance_method:
                        cluster_method_name = ood_method.method1.cluster_method
                        mean_n_clusters, std_n_clusters = get_mean_and_std_n_clusters(ood_method.method1)
                    elif ood_method.method2.is_distance_method:
                        cluster_method_name = ood_method.method2.cluster_method
                        mean_n_clusters, std_n_clusters = get_mean_and_std_n_clusters(ood_method.method2)
                    else:
                        raise ValueError("At least one of the methods must be a distance method if indicated in the base class")
                else:
                    cluster_method_name = 'No Cluster Method'
                    mean_n_clusters, std_n_clusters = 0, 0

                fill_dict_with_method_info(results_one_run, args, ood_method, cluster_method=cluster_method_name, mean_n_clus=mean_n_clusters, std_n_clus=std_n_clusters)

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


if __name__ == "__main__":
    main(SimpleArgumentParser().parse_args())


# elif args.benchmark == 'fusion_strategies':
#             ## 1. Name results file
#             results_file_name = f"{NOW}_{args.benchmark}_{args.ood_method}_conf_train{args.conf_thr_train}_conf_test{args.conf_thr_test}"
#             FUSION_STRATS_TO_TEST = BENCHMARKS[args.benchmark]
#             logger.info(f"Running benchmark for methods {FUSION_STRATS_TO_TEST}")

#             ## 2. Load common assets for the benchmark if any
#             logger.info(f"First configure and train the OOD detection method")
#             # Load the OOD detection method
#             ood_method = select_ood_detection_method(args)
#             # Modify internal attributes of the model to obtain the desired outputs in the extra_item
#             configure_extra_output_of_the_model(model, ood_method)
#             execute_pipeline_for_in_distribution_configuration(ood_method, model, device, ind_dataloader, ind_val_dataloader, logger, args)
            
#             ## 3. Run the benchmark
#             final_results_df = pd.DataFrame(columns=results_colums)
#             for strat in FUSION_STRATS_TO_TEST:
#                 print("-"*50)
#                 results_one_run = {}
#                 ## 3.1. Modify what is going to be benchmarked
#                 logger.info(f" *** Fusion strategy: {strat} ***")
#                 # Change to the new strat
#                 args.fusion_strategy = strat
#                 ood_method.fusion_strategy = strat
                
#                 ## 3.2. Add info
#                 if ood_method.is_distance_method:
#                     if ood_method.method1.is_distance_method and ood_method.method2.is_distance_method:
#                         cluster_method_name = f"{ood_method.method1.cluster_method} + {ood_method.method2.cluster_method}"
#                         mean_n_clusters1, std_n_clusters1 = get_mean_and_std_n_clusters(ood_method.method1)
#                         mean_n_clusters2, std_n_clusters2 = get_mean_and_std_n_clusters(ood_method.method2)
#                         mean_n_clusters = (mean_n_clusters1 + mean_n_clusters2) / 2
#                         std_n_clusters = (std_n_clusters1 + std_n_clusters2) / 2
#                     elif ood_method.method1.is_distance_method:
#                         cluster_method_name = ood_method.method1.cluster_method
#                         mean_n_clusters, std_n_clusters = get_mean_and_std_n_clusters(ood_method.method1)
#                     elif ood_method.method2.is_distance_method:
#                         cluster_method_name = ood_method.method2.cluster_method
#                         mean_n_clusters, std_n_clusters = get_mean_and_std_n_clusters(ood_method.method2)
#                     else:
#                         raise ValueError("At least one of the methods must be a distance method if indicated in the base class")
#                 else:
#                     cluster_method_name = 'No Cluster Method'
#                     mean_n_clusters, std_n_clusters = 0, 0

#                 results_one_run['Method'] = args.ood_method
#                 results_one_run['conf_thr_train'] = args.conf_thr_train
#                 results_one_run['conf_thr_test'] = args.conf_thr_test
#                 results_one_run["tpr_thr"] = args.tpr_thr
#                 results_one_run["cluster_method"] = cluster_method_name
#                 results_one_run["mean_n_clus"] = mean_n_clusters
#                 results_one_run["std_n_clus"] = std_n_clusters
#                 # results_one_run["mean_num_samples_per_clus"] = mean_num_samples_per_clus
#                 # results_one_run["std_num_samples_per_clus"] = std_num_samples_per_clus
#                 results_one_run["fusion_strat"] = args.fusion_strategy

#                 ## 3.3. Run configuration for every dataset
#                 for dataloader in ood_dataloaders:
#                     # Extract metrics
#                     results_one_run_one_dataset = run_eval(ood_method, model, device, dataloader, known_classes, logger)
#                     if coco_ood_dataloader == dataloader:
#                         dataset_name = COCO_OOD_NAME    
#                     elif coco_mixed_dataloader == dataloader:
#                         dataset_name = COCO_MIXED_NAME
#                     elif coco_owod_test_dataloader == dataloader:
#                         dataset_name = COCO_OWOD_TEST_NAME
#                     else:
#                         raise ValueError("Unknown dataset")
#                     fill_dict_with_one_dataset_results(results_one_run, results_one_run_one_dataset, dataset_name)
                
#                 ## 3.4. Collect results
#                 add_args_and_hyperparams_info(results_one_run, args, CUSTOM_HYP)
#                 final_results_df.loc[len(final_results_df)] = results_one_run
#                 print("-"*50, '\n')