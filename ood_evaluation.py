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

from ood_utils import configure_extra_output_of_the_model, OODMethod, LogitsMethod, DistanceMethod, MSP, Energy, ODIN, Sigmoid, \
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
    ood_method: str  # OOD detection method to use. If it is a fusion method, it must be passed as 'fusion-logits_method-distance_method'.
    visualize_oods: bool = False  # visualize the OoD detection
    compute_metrics: bool = False  # compute the metrics
    benchmark: Literal['', 'best_methods', 'conf_thr_test', 'cluster_methods', 'logits_methods'] = ''  # Benchmark to run
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
    benchmark_datasets: List[str] = []  # Datasets to use for the benchmark. Options: 'coco_ood', 'coco_mixed', 'coco_owod_test'
    # Hyperparameters for YOLO
    conf_thr_train: float = 0.15  # Confidence threshold for the In-Distribution configuration
    conf_thr_test: float = 0.15  # Confidence threshold for the detections
    # Hyperparameters for the OOD detection
    tpr_thr: float = 0.95  # TPR threshold for the OoD detection
    cluster_method: str = 'one'  # Clustering method to use for the distance methods
    cluster_optimization_metric: Literal['silhouette', 'calinski_harabasz'] = 'silhouette'  # Metric to use for the optimization of the clusters
    ind_info_creation_option: str = 'valid_preds_one_stride'  # How to create the in-distribution information for the distance methods
    enhanced_unk_localization: bool = False  # Whether to use enhanced unknown localization
    which_internal_activations: str = 'roi_aligned_ftmaps'  # Which internal activations to use for the OoD detection
    # Hyperparams for methods
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
        self.add_argument('--ood_method', choices=OOD_METHOD_CHOICES, required=True, help='OOD detection method to use')
        self.add_argument(
            '--benchmark_datasets', 
            nargs='+', 
            choices=[COCO_OOD_NAME, COCO_MIXED_NAME, COCO_OWOD_TEST_NAME], 
            help="Datasets to use for the benchmark"
        )

    def process_args(self):

        if self.model_path:
            print('Loading model from', self.model_path)
            print('Ignoring args --model --from_scratch')
            self.from_scratch = False
            self.model = ''
        else:
            if self.model == '':
                raise ValueError("You must pass a model size.")
        
        if not self.visualize_oods and not self.compute_metrics and not self.benchmark:
            raise ValueError("You must pass either visualize_oods or compute_metrics or define a benchmark")
        
        if self.benchmark and not len(self.benchmark_datasets) > 0:
            raise ValueError("You must pass benchmark_datasets to run a benchmark")
        
        if self.benchmark == 'cluster_methods':
            if self.ood_method not in DISTANCE_METHODS:
                raise ValueError("You must select a distance method to run this benchmark")



def select_ood_detection_method(args: SimpleArgumentParser) -> Union[LogitsMethod, DistanceMethod]:
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
        _, logits_method, distance_method = args.ood_method.split('-')
        args.ood_method = logits_method
        ood_method_logits = select_ood_detection_method(args)
        args.ood_method = distance_method
        ood_method_distance = select_ood_detection_method(args)
        return FusionMethod(ood_method_logits, ood_method_distance, args.fusion_strategy)

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


def execute_pipeline_for_in_distribution_configuration(ood_method: Union[LogitsMethod, DistanceMethod], model: YOLO, device: str, 
                                               in_loader: InfiniteDataLoader, logger: Logger, args: SimpleArgumentParser):
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
    activations_str =   f'{ood_method.which_internal_activations}_conf{ood_method.min_conf_threshold_train}_{model.ckpt["train_args"]["name"]}_activations'
    thresholds_str =    f'{ood_method.name}_conf{ood_method.min_conf_threshold_train}_{model.ckpt["train_args"]["name"]}_thresholds'
    if args.ood_method in DISTANCE_METHODS:
        clusters_str = f'{ood_method.name}_conf{ood_method.min_conf_threshold_train}_{model.ckpt["train_args"]["name"]}_clusters_{ood_method.cluster_method}_{ood_method.cluster_optimization_metric}'
        thresholds_str += f'_{ood_method.cluster_method}'
    if args.ood_method in TARGETS_RELATED_OPTIONS:
        activations_str += f'_{args.ind_info_creation_option}'
        thresholds_str += f'_{args.ind_info_creation_option}'
        if args.ood_method in DISTANCE_METHODS:
            clusters_str += f'_{args.ind_info_creation_option}'
    
    activations_path = STORAGE_PATH / f'{activations_str}.pt'
    thresholds_path = STORAGE_PATH / f'{thresholds_str}.json'
    if args.ood_method in DISTANCE_METHODS:
        clusters_path = STORAGE_PATH / f'{clusters_str}.pt'

    # if args.ind_info_creation_option in TARGETS_RELATED_OPTIONS:
    #     activations_path = STORAGE_PATH / f'{ood_method.which_internal_activations}_{model.ckpt["train_args"]["name"]}_activations_{args.ind_info_creation_option}.pt'
    #     if args.ood_method in DISTANCE_METHODS:
    #         clusters_path = STORAGE_PATH / f'{ood_method.name}_{model.ckpt["train_args"]["name"]}_clusters_{ood_method.cluster_method}_{args.ind_info_creation_option}.pt'
    #         thresholds_path = STORAGE_PATH / f'{ood_method.name}_{model.ckpt["train_args"]["name"]}_thresholds_{ood_method.cluster_method}_{args.ind_info_creation_option}.json'
    #     else:
    #         thresholds_path = STORAGE_PATH / f'{ood_method.name}_{model.ckpt["train_args"]["name"]}_thresholds_{args.ind_info_creation_option}.json'
    # else:
    #     activations_path = STORAGE_PATH / f'{ood_method.which_internal_activations}_{model.ckpt["train_args"]["name"]}_activations.pt'
    #     if args.ood_method in DISTANCE_METHODS:
    #         clusters_path = STORAGE_PATH / f'{ood_method.name}_{model.ckpt["train_args"]["name"]}_clusters_{ood_method.cluster_method}.pt'
    #         thresholds_path = STORAGE_PATH / f'{ood_method.name}_{model.ckpt["train_args"]["name"]}_thresholds_{ood_method.cluster_method}.json'
    #     else:
    #         thresholds_path = STORAGE_PATH / f'{ood_method.name}_{model.ckpt["train_args"]["name"]}_thresholds.json'

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
        # Load activations for the thresholds
        if args.load_ind_activations:
            # Load in_distribution activations from disk
            logger.info("Loading in-distribution activations...")
            if activations_path.exists():
                ind_activations = torch.load(activations_path)
                logger.info(f"In-distribution activations succesfully loaded from {activations_path}")
            else:
                # Generate in_distribution activations to generate thresholds
                logger.error(f"File {activations_path} does not exist. Generating in-distribution activations by iterating over the data...")
                ind_activations = ood_method.iterate_data_to_extract_ind_activations(in_loader, model, device, logger)
                logger.info("In-distribution data processed")
                logger.info("Saving in-distribution activations...")
                torch.save(ind_activations, activations_path, pickle_protocol=5)
                logger.info(f"In-distribution activations succesfully saved in {activations_path}")

        # Generate in_distribution activations to generate thresholds
        else:
            logger.info("Processing in-distribution data...")
            ind_activations = ood_method.iterate_data_to_extract_ind_activations(in_loader, model, device, logger)
            logger.info("In-distribution data processed")
            logger.info("Saving in-distribution activations...")
            torch.save(ind_activations, activations_path, pickle_protocol=5)
            logger.info(f"In-distribution activations succesfully saved in {activations_path}")

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
                    ood_method.clusters = ood_method.generate_clusters(ind_activations, logger)
                    logger.info("Saving clusters...")
                    torch.save(ood_method.clusters, clusters_path, pickle_protocol=5)
                    logger.info(f"Clusters succesfully saved in {clusters_path}")
                logger.info(f"Clusters succesfully loaded from {clusters_path}")

            # Generate the clusters using the In-Distribution activations
            else:
                # Generate in_distribution clusters to generate thresholds for OOD method
                logger.info("Generating clusters...")
                ood_method.clusters = ood_method.generate_clusters(ind_activations, logger)
                logger.info("Saving clusters...")
                torch.save(ood_method.clusters, clusters_path, pickle_protocol=5)

            # Generate the scores that are necessary to create the thresholds by using the clusters and the activations
            logger.info("Generating in-distribution scores...")
            ind_scores = ood_method.compute_scores_from_activations(ind_activations, logger)

            if hasattr(CUSTOM_HYP.unk, 'rank'):
                if CUSTOM_HYP.unk.rank.USE_UNK_PROPOSALS_THR:
                    logger.info("Generating scores to evaluate UNK proposals...")
                    scores_for_unk_prop = ood_method.compute_scores_from_activations_for_unk_proposals(ind_activations, logger)
                    logger.info("Saving UNK proposals...")
        
        # For the rest of the methods activations are the scores themselves
        else:
            ind_scores = ood_method.compute_scores_from_activations(ind_activations, logger)

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
        write_json(ood_method.thresholds, thresholds_path)


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
        
        ### OOD evaluation ###

        # Main function that executes the pipeline for the OOD evaluation (explained inside the function)
        execute_pipeline_for_in_distribution_configuration(ood_method, model, device, ind_dataloader, logger, args)

        if args.visualize_oods:    
            # Save images with OoD detection (Green for In-Distribution, Red for Out-of-Distribution, Violet the Ground Truth)
            save_images_with_ood_detection(ood_method, model, device, ood_dataloader, logger)
            
        elif args.compute_metrics:
            
            # Run the normal evaluation to compute the metrics
            _ = run_eval(ood_method, model, device, ood_dataloader, known_classes, logger)
        
        else:
            raise ValueError("You must pass either visualize_oods or compute_metrics")
        
        end_time = time.time()
        logger.info("Total running time: {}".format(end_time - start_time))
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
        if args.benchmark == 'best_methods':
            raise NotImplementedError("Not implemented yet")

        elif args.benchmark == 'conf_thr_test':
            ## 1. Name results file
            if args.ood in DISTANCE_METHODS:
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
            execute_pipeline_for_in_distribution_configuration(ood_method, model, device, ind_dataloader, logger, args)

            ## 3. Run the benchmark
            final_results_df = pd.DataFrame(columns=results_colums)
            for conf_threshold in CONF_THR_TEST_BENCHMARK:
                results_one_run = {}
                print("-"*50)
                ## 3.1. Modify what is going to be benchmarked
                logger.info(f" *** Confidence threshold: {conf_threshold} ***")
                ood_method.min_conf_threshold_test = conf_threshold

                ## 3.2. Add info
                results_one_run['Method'] = args.ood_method
                results_one_run['Conf_threshold'] = conf_threshold
                results_one_run["tpr_thr"] = args.tpr_thr
                results_one_run["cluster_method"] = ood_method.cluster_method

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
                    fill_dict_with_results(results_one_run, results_one_run_one_dataset, dataset_name)
                
                ## 3.4. Collect results
                add_args_and_hyperparams_info(results_one_run, args, CUSTOM_HYP)
                final_results_df.loc[len(final_results_df)] = results_one_run
                print("-"*50, '\n')

        ###
        ##
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
            
            ## 3. Run the benchmark
            final_results_df = pd.DataFrame(columns=results_colums)
            for cluster_method_one_run in CLUSTER_METHODS_TO_TEST:
                print("-"*50)
                results_one_run = {}
                ## 3.1. Modify what is going to be benchmarked
                logger.info(f"*** Cluster method: {cluster_method_one_run} ***")
                ood_method.cluster_method = cluster_method_one_run
                # Create all the info for the configuration of the OOD detection method
                execute_pipeline_for_in_distribution_configuration(ood_method, model, device, ind_dataloader, logger, args)
                
                ## 3.2. Add info
                results_one_run['Method'] = args.ood_method
                results_one_run['Conf_threshold'] = args.conf_thr_test
                results_one_run["tpr_thr"] = args.tpr_thr
                results_one_run["cluster_method"] = ood_method.cluster_method

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
                    fill_dict_with_results(results_one_run, results_one_run_one_dataset, dataset_name)
                
                ## 3.4. Collect results
                add_args_and_hyperparams_info(results_one_run, args, CUSTOM_HYP)
                final_results_df.loc[len(final_results_df)] = results_one_run
                print("-"*50, '\n')
        
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
                execute_pipeline_for_in_distribution_configuration(ood_method, model, device, ind_dataloader, logger, args)
                
                ## 3.2. Add info
                results_one_run['Method'] = args.ood_method
                results_one_run['Conf_threshold'] = args.conf_thr_test
                results_one_run["tpr_thr"] = args.tpr_thr
                results_one_run["cluster_method"] = ood_method.cluster_method

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
                    fill_dict_with_results(results_one_run, results_one_run_one_dataset, dataset_name)
                
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
        logger.info("Total running time: {}".format(global_end_time - global_start_time))


def fill_dict_with_results(results_dict: Dict[str, float], results_one_dataset: Dict[str, float], dataset_name: str) -> None:
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


if __name__ == "__main__":
    main(SimpleArgumentParser().parse_args())
