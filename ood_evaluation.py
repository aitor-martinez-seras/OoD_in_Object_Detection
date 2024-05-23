import time
import os
from pathlib import Path
from datetime import datetime
from typing import Type, Union, Literal, List, Tuple, Dict
from logging import Logger

from tap import Tap

#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

import log
from ultralytics import YOLO
from ultralytics.yolo.data import build_dataloader
from ultralytics.yolo.data import BaseDataset
from ultralytics.yolo.data.build import InfiniteDataLoader


from ood_utils import get_measures, configure_extra_output_of_the_model, OODMethod, LogitsMethod, DistanceMethod, MSP, Energy, \
    L1DistanceOneClusterPerStride, L2DistanceOneClusterPerStride, GAPL2DistanceOneClusterPerStride, CosineDistanceOneClusterPerStride
from data_utils import read_json, write_json, load_dataset_and_dataloader
from unknown_localization_utils import select_ftmaps_summarization_method, select_thresholding_method
from constants import ROOT, STORAGE_PATH, PRUEBAS_ROOT_PATH, RESULTS_PATH, OOD_METHOD_CHOICES, CONF_THRS_FOR_BENCHMARK
from custom_hyperparams import CUSTOM_HYP


NOW = datetime.now().strftime("%Y%m%d_%H%M%S")

class SimpleArgumentParser(Tap):
    # MAIN OPTION TO SELECT
    visualize_oods: bool = False  # visualize the OoD detection
    compute_metrics: bool = False  # compute the metrics
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
    benchmark_conf: bool = False  # Run confidence benchmark over confidence thresholds [0.15, 0.10, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.00001]
    # Hyperparameters
    conf_thr: float = 0.15  # Confidence threshold for the detections
    tpr_thr: float = 0.95  # TPR threshold for the OoD detection
    # Datasets
    ind_dataset: str  # Dataset to use for training and validation
    ind_split: Literal['train', 'val', 'test'] = 'train'  # Split to use in the in-distribution dataset
    ood_dataset: str  # Dataset to use for OoD detection
    ood_split: Literal['train', 'val', 'test'] = 'val'  # Split to use in the out-of-distribution dataset
    owod_task_ind: Literal["", "t1", "t2", "t3", "t4", "all_task_test"] = ""  # OWOD task to use in the in-distribution dataset
    owod_task_ood: Literal["", "t1", "t2", "t3", "t4", "all_task_test"] = ""  # OWOD task to use in the out-of-distribution dataset
    # OOD related
    ood_method: str
    ind_info_creation_option: str = 'valid_preds_one_stride'  # How to create the in-distribution information for the distance methods
    enhanced_unk_localization: bool = False  # Whether to use enhanced unknown localization
    which_internal_activations: str = 'roi_aligned_ftmaps'  # Which internal activations to use for the OoD detection
    # ODIN and Energy
    temperature: int = 1000
    epsilon_odin: float = 0.0
    load_ind_activations: bool = False  # load in-distribution scores from disk
    load_clusters: bool = False  # load clusters from disk
    load_thresholds: bool = False  # load thresholds from disk
    
    def configure(self):
        self.add_argument("-m", "--model", required=False)
        self.add_argument('--ood_method', choices=OOD_METHOD_CHOICES, required=True, help='OOD detection method to use')

    def process_args(self):

        if self.model_path:
            print('Loading model from', self.model_path)
            print('Ignoring args --model --from_scratch')
            self.from_scratch = False
            self.model = ''
        else:
            if self.model == '':
                raise ValueError("You must pass a model size.")
        
        if not self.visualize_oods and not self.compute_metrics:
            raise ValueError("You must pass either visualize_oods or compute_metrics")


def select_ood_detection_method(args: SimpleArgumentParser) -> Union[LogitsMethod, DistanceMethod]:
    """
    Select the OOD method to use for the evaluation.
    """
    common_kwargs = {
        'iou_threshold_for_matching': CUSTOM_HYP.IOU_THRESHOLD,
        'min_conf_threshold': args.conf_thr
    }
    distance_methods_kwargs = {
        'agg_method': 'mean',
        'ind_info_creation_option': args.ind_info_creation_option,
        'enhanced_unk_localization': args.enhanced_unk_localization,
        'which_internal_activations': args.which_internal_activations,
        'saliency_map_computation_function': select_ftmaps_summarization_method(CUSTOM_HYP.unk.SUMMARIZATION_METHOD),
        'thresholds_out_of_saliency_map_function': select_thresholding_method(CUSTOM_HYP.unk.THRESHOLDING_METHOD),
    }
    distance_methods_kwargs.update(common_kwargs)

    if args.ood_method == 'MSP':
        return MSP(per_class=True, per_stride=False, **common_kwargs)
    elif args.ood_method == 'Energy':
        return Energy(temper=args.temperature, per_class=True, per_stride=False, **common_kwargs)
    elif args.ood_method == 'L1_cl_stride':
        #return L1DistanceOneClusterPerStride(agg_method='mean', iou_threshold_for_matching=IOU_THRESHOLD, min_conf_threshold=args.conf_thr)
        return L1DistanceOneClusterPerStride(**distance_methods_kwargs)
    elif args.ood_method == 'L2_cl_stride':
        #return L2DistanceOneClusterPerStride(agg_method='mean', iou_threshold_for_matching=IOU_THRESHOLD, min_conf_threshold=args.conf_thr)
        return L2DistanceOneClusterPerStride(**distance_methods_kwargs)
    elif args.ood_method == 'GAP_L2_cl_stride':
        #return GAPL2DistanceOneClusterPerStride(agg_method='mean', iou_threshold_for_matching=IOU_THRESHOLD, min_conf_threshold=args.conf_thr)
        return GAPL2DistanceOneClusterPerStride(**distance_methods_kwargs)
    elif args.ood_method == 'Cosine_cl_stride':
        #return CosineDistanceOneClusterPerStride(agg_method='mean', iou_threshold_for_matching=IOU_THRESHOLD, min_conf_threshold=args.conf_thr)
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

    thresholds_path = STORAGE_PATH / f'{ood_method.name}_{model.ckpt["train_args"]["name"]}_thresholds.json'
    activations_path = STORAGE_PATH / f'{ood_method.which_internal_activations}_{model.ckpt["train_args"]["name"]}_activations.pt'
    clusters_path = STORAGE_PATH / f'{ood_method.name}_{model.ckpt["train_args"]["name"]}_clusters.pt'

    ### Load the thresholds ###
    if args.load_thresholds:
        # Load thresholds from disk
        ood_method.thresholds = read_json(thresholds_path)
        logger.info(f"Thresholds succesfully loaded from {thresholds_path}")
        # For a distance method also the clusters are needed
        if ood_method.distance_method:
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
            ind_activations = torch.load(activations_path)
            logger.info(f"In-distribution activations succesfully loaded from {activations_path}")

        # Generate in_distribution activations to generate thresholds
        else:
            logger.info("Processing in-distribution data...")
            ind_activations = ood_method.iterate_data_to_extract_ind_activations(in_loader, model, device, logger)
            logger.info("In-distribution data processed")
            logger.info("Saving in-distribution activations...")
            # Save activations
            torch.save(ind_activations, STORAGE_PATH / f'{ood_method.which_internal_activations}_{model.ckpt["train_args"]["name"]}_activations.pt', pickle_protocol=5)
            logger.info(f"In-distribution activations succesfully saved in {activations_path}")

        ### 2. Obtain scores ###
        # Distance methods need to have clusters representing the In-Distribution data and then compute the scores
        if ood_method.distance_method:
            
            ### 2.1. Distance methods need to obtain clusters for scores ###
            # Load the clusters
            if args.load_clusters:
                # Load in_distribution clusters from disk
                ood_method.clusters = torch.load(clusters_path)

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

            # if unk_prop_thr:
            #     logger.info("Generating scores to evaluate UNK proposals...")
            #     scores_for_unk_prop = ood_method.compute_scores_from_activations_for_unk_prop(ind_activations, logger)
            #     logger.info("Saving UNK proposals...")
        
        # For the rest of the methods activations are the scores themselves
        else:
            ind_scores = ind_activations 

        ### 3. Obtain thresholds ###
        # Finally generate and save the thresholds
        logger.info("Generating thresholds...")
        ood_method.thresholds = ood_method.generate_thresholds(ind_scores, tpr=args.tpr_thr, logger=logger)
        # if unk_prop_thr:
            #     logger.info("Generating scores to evaluate UNK proposals...")
            #     scores_for_unk_prop = ood_method.genera(ind_activations, logger)
            #     logger.info("Saving UNK proposals...")
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

    # Load datasets
    ind_dataset, ind_dataloader = load_dataset_and_dataloader(
        dataset_name=args.ind_dataset,
        data_split=args.ind_split,
        batch_size=args.batch_size,
        workers=args.workers,
        owod_task=args.owod_task_ind
    )
    ood_dataset, ood_dataloader = load_dataset_and_dataloader(
        dataset_name=args.ood_dataset,
        data_split=args.ood_split,
        batch_size=args.batch_size,
        workers=args.workers,
        owod_task=args.owod_task_ood
    )

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

    logger.info(f"IoU threshold set to {CUSTOM_HYP.IOU_THRESHOLD}")

    # Load the OOD detection method
    ood_method = select_ood_detection_method(args)

    # Modify internal attributes of the model to obtain the desired outputs in the extra_item
    configure_extra_output_of_the_model(model, ood_method)            

    start_time = time.time()
    
    ### OOD evaluation ###
    # TODO: Si metemos otro dataset habra que hacer esto de forma mas general
    known_classes = [x for x in range(ind_dataset.number_of_classes)]

    # Main function that executes the pipeline for the OOD evaluation (explained inside the function)
    execute_pipeline_for_in_distribution_configuration(ood_method, model, device, ind_dataloader, logger, args)

    if args.visualize_oods:    
        # Save images with OoD detection (Green for In-Distribution, Red for Out-of-Distribution, Violet the Ground Truth)
        save_images_with_ood_detection(ood_method, model, device, ood_dataloader, logger)
        
    elif args.compute_metrics:
        
        ### Compute the metrics for various confidence thresholds ###
        if args.benchmark_conf:
            logger.info(f"Running benchmark for confidences {CONF_THRS_FOR_BENCHMARK} in datasets {args.ind_dataset} vs {args.ood_dataset}")
            import pandas as pd
            final_results = []
            for conf_threshold in CONF_THRS_FOR_BENCHMARK:
                print("-"*50)
                logger.info(f" *** Confidence threshold: {conf_threshold} ***")
                ood_method.min_conf_threshold = conf_threshold
                # Extract metrics
                results_one_run = run_eval(ood_method, model, device, ood_dataloader, known_classes, logger)
                # Make data be in range [0, 1]
                for key in results_one_run.keys():
                    if key not in ["A-OSE", "WI-08"]:
                        results_one_run[key] = results_one_run[key] / 100
                res_columns = list(results_one_run.keys())
                results_one_run['Method'] = args.ood_method
                results_one_run['Conf_threshold'] = conf_threshold
                results_one_run["tpr_thr"] = args.tpr_thr
                results_one_run['Model'] = args.model_path if args.model_path else model_to_load
                
                final_results.append(results_one_run)
                print("-"*50, '\n')
            # Save the results
            results_df = pd.DataFrame(final_results)
            results_df = results_df[['Method', 'Conf_threshold', 'tpr_thr'] + res_columns + ['Model']]  # Reorder columns
            results_df.to_csv(RESULTS_PATH / f'{NOW}_{args.ood_method}.csv', index=False)
            results_df.to_excel(RESULTS_PATH / f'{NOW}_{args.ood_method}.xlsx', index=False)

        ### Compute metrics for a single configuration ###
        else:
            # Run the normal evaluation to compute the metrics
            _ = run_eval(ood_method, model, device, ood_dataloader, known_classes, logger)

    end_time = time.time()

    logger.info("Total running time: {}".format(end_time - start_time))
    logger.info(CUSTOM_HYP)


if __name__ == "__main__":
    main(SimpleArgumentParser().parse_args())
