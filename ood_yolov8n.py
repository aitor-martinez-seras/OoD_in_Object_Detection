import time
import os
from pathlib import Path
from datetime import datetime
from typing import Type
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

import log
from ultralytics import YOLO
from ultralytics.yolo.data import build_dataloader

from ood_utils import get_measures, configure_extra_output_of_the_model, OODMethod, LogitsMethod, DistanceMethod, MSP, Energy, ODIN, \
    L1DistanceOneClusterPerStride, L2DistanceOneClusterPerStride, GAPL2DistanceOneClusterPerStride

from data_utils import read_json, write_json, create_YOLO_dataset_and_dataloader, build_dataloader, create_TAO_dataset_and_dataloader


NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
STORAGE_PATH = Path('storage')
PRUEBAS_ROOT_PATH = Path('pruebas')

IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.15

OOD_METHOD_CHOICES = ['MSP', 'ODIN', 'Energy', 'Mahalanobis', 'GradNorm','RankFeat','React', 'L1_cl_stride', 'L2_cl_stride', \
                      'GAP_L2_cl_stride']


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--workers", type=int, default=0,
                        help="Number of background threads used to load data.")

    parser.add_argument("--logdir", default='logs',
                        help="Where to log test info (small).")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size.")
    parser.add_argument("--name", default='prueba',
                        help="Name of this run. Used for monitoring and checkpointing.")

    parser.add_argument("--model", default="YOLO", help="Which variant to use")
    parser.add_argument("--model_path", type=str, help="Path to the model you want to test")

    return parser


def select_ood_detection_method(args) -> Type[OODMethod]:
    """
    Select the OOD method to use for the evaluation.
    """
    if args.ood_method == 'MSP':
        return MSP(per_class=True, per_stride=False, iou_threshold_for_matching=IOU_THRESHOLD, min_conf_threshold=CONF_THRESHOLD)
    elif args.ood_method == 'ODIN':
        return ODIN()
    elif args.ood_method == 'Energy':
        return Energy(temper=args.temperature, per_class=True, per_stride=False, iou_threshold_for_matching=IOU_THRESHOLD, min_conf_threshold=CONF_THRESHOLD)
    elif args.ood_method == 'L1_cl_stride':
        return L1DistanceOneClusterPerStride(agg_method='mean', iou_threshold_for_matching=IOU_THRESHOLD, min_conf_threshold=CONF_THRESHOLD)
    elif args.ood_method == 'L2_cl_stride':
        return L2DistanceOneClusterPerStride(agg_method='mean', iou_threshold_for_matching=IOU_THRESHOLD, min_conf_threshold=CONF_THRESHOLD)
    elif args.ood_method == 'GAP_L2_cl_stride':
        return GAPL2DistanceOneClusterPerStride(agg_method='mean', iou_threshold_for_matching=IOU_THRESHOLD, min_conf_threshold=CONF_THRESHOLD)
    else:
        raise NotImplementedError("Not implemented yet")




def obtain_thresholds_for_ood_detection_method(ood_method: Type[OODMethod], model, device, in_loader, logger, args):
    """
    Function that loads or generates the thresholds for the OOD evaluation. The thresholds are
        stored in the OODMethod object.
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
        # Distance methods need to have clusters representing the In-Distribution data
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

        else:  # For the rest of the methods activations are the scores themselves
            ind_scores = ind_activations 

        ### 3. Obtain thresholds ###
        # Finally generate and save the thresholds
        logger.info("Generating thresholds...")
        ood_method.thresholds = ood_method.generate_thresholds(ind_scores, tpr=0.95, logger=logger)
        logger.info("Saving thresholds...")
        write_json(ood_method.thresholds, thresholds_path)


def save_images_with_ood_detection(ood_method: OODMethod, model, device, ood_loader, logger):

    assert ood_method.thresholds is not None, "Thresholds must be generated or loaded before predicting with OoD detection"

    logger.info("Predicting with OOD detection...")
    
    ood_method.iterate_data_to_plot_with_ood_labels(model, ood_loader, device, logger, PRUEBAS_ROOT_PATH, NOW)


def run_eval(ood_method: OODMethod, model, device, in_loader, ood_loader, logger, args):
    logger.info("Running test to compute metrics...")
    logger.flush()


def main(args):
    print('----------------------------')
    print('****************************')
    print('****************************')
    print('****************************')
    
    # Setup logger
    logger = log.setup_logger(args)

    # TODO: This is for reproducibility 
    # torch.backends.cudnn.benchmark = True

    # logger.warning('Changing following enviroment variables:')
    # os.environ['YOLO_VERBOSE'] = 'False'
    gpu_number = str(args.device)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_number
    logger.warning(f'CUDA_VISIBLE_DEVICES = {gpu_number}')
    device = 'cuda:0'

    if args.ood_method == 'GradNorm':
        args.batch = 1
        logger.warning(f'Batch size changed to {args.batch} as using GradNorm')

    # Load ID data and OOD data
    # TODO: Aqui tengo que meter algo que compruebe que el dataset esta como YAML file
    #if yaml_file_exists('coco.yaml'):
    

    ### OAK dataset ###
    if False:
        ind_dataset, ind_dataloader = create_YOLO_dataset_and_dataloader(
            'OAK_full.yaml',
            args,
            data_split='train',
        )
        if False:
            ood_dataset, ood_dataloader = create_YOLO_dataset_and_dataloader(
                'VisDrone.yaml', 
                args=args,
                data_split='val',
            )
        else:
            # ood_dataset = SOS_BaseDataset(
            #     imgs_path='/home/tri110414/nfs_home/datasets/street_obstacle_sequences/raw_data/',
            #     ann_path='/home/tri110414/nfs_home/datasets/street_obstacle_sequences/val_annotations.json',
            #     imgsz=640
            # )

            ood_dataset = OAKDataset(
                imgs_path='/home/tri110414/nfs_home/datasets/OAK/val/Raw',
                ann_path='/home/tri110414/nfs_home/datasets/OAK/val/val_annotations_coco.json',
                imgsz=1152
            )

            ood_dataloader = build_dataloader(
                ood_dataset,
                batch=args.batch_size,
                workers=args.num_workers,
                shuffle=False,
                rank=-1
            )

    ### TAO dataset ###
    if True:
        # ind_dataset = TAODataset(
        #     imgs_path='/home/tri110414/nfs_home/datasets/TAO/frames/train',
        #     ann_path='/home/tri110414/nfs_home/datasets/TAO/annotations/train.json',
        #     imgsz=1152
        # )

        # ind_dataloader = build_dataloader(
        #     ind_dataset,
        #     batch=args.batch_size,
        #     workers=args.num_workers,
        #     shuffle=False,
        #     rank=-1
        # )

        # # ood_dataset = TAODataset(
        # #     imgs_path='/home/tri110414/nfs_home/datasets/TAO/frames/train',
        # #     ann_path='/home/tri110414/nfs_home/datasets/TAO/annotations/train.json',
        # #     imgsz=1152
        # # )

        # ood_dataloader = build_dataloader(
        #     ind_dataset,  # TODO: Change this to the OOD dataset
        #     batch=args.batch_size,
        #     workers=args.num_workers,
        #     shuffle=False,
        #     rank=-1
        # )

        ind_dataset, ind_dataloader = create_TAO_dataset_and_dataloader(
            'tao_coco.yaml',
            args,
            data_split='train',
        )

        ood_dataset, ood_dataloader = create_TAO_dataset_and_dataloader(
            'tao_coco.yaml',
            args,
            data_split='train',
        )
    else:
        # COCO
        ind_dataset, ind_dataloader = create_YOLO_dataset_and_dataloader(
            'coco.yaml',
            args,
            data_split='train',
        )

        ood_dataset, ood_dataloader = create_YOLO_dataset_and_dataloader(
            'coco.yaml',
            args,
            data_split='val',
        )


    # TODO: usar el argparser para elegir el modelo que queremos cargar
    model_to_load = 'yolov8n.pt'

    logger.info(f"Loading model {model_to_load} in {args.device}")

    logger.info(f"IoU threshold set to {IOU_THRESHOLD}")

    # Load YOLO model
    # TODO: add different YOLO models
    model = YOLO(model_to_load) 
    # state_dict = torch.load(args.model_path)
    # model.load_state_dict_custom(state_dict['model'])

    # TODO: Unused till we implement GradNorm
    # if args.ood_method != 'GradNorm':
    #     model = torch.nn.DataParallel(model)

    # Load the OOD detection method
    ood_method = select_ood_detection_method(args)

    # Modify internal attributes of the model to obtain the desired outputs in the extra_item
    configure_extra_output_of_the_model(model, ood_method)            

    start_time = time.time()
    if args.visualize_oods:
        
        # First fill the thresholds attribute of the OODMethod object
        obtain_thresholds_for_ood_detection_method(ood_method, model, device, ind_dataloader, logger, args)
        
        # Save images with OoD detection (Green for In-Distribution, Red for Out-of-Distribution)
        save_images_with_ood_detection(ood_method, model, device, ood_dataloader, logger)
        
    elif args.compute_metrics:

        # Esta será la funcion que saque las metricas AUROC, AUPR, FPR95
        run_eval(model, args.device, ind_dataloader, ood_dataloader, logger, args)

    end_time = time.time()

    logger.info("Total running time: {}".format(end_time - start_time))


if __name__ == "__main__":
    parser = arg_parser()
    
    parser.add_argument('-d', '--device', default=0, type=int, help='-1 for cpu or a number for the index of the GPU to use')
    parser.add_argument('-b', '--batch_size', default=4, type=int, help='batch size to use')
    parser.add_argument('-n_w', '--num_workers', default=1, type=int, help='number of workers to use in dataloader')
    parser.add_argument('-v', '--visualize_oods', action='store_true', help='visualize the OoD detection')
    parser.add_argument('--load_ind_activations', action='store_true', help='load in-distribution scores from disk')
    parser.add_argument('--load_clusters', action='store_true', help='load clusters from disk')
    parser.add_argument('--load_thresholds', action='store_true', help='load thresholds from disk')
    # parser.add_argument("--in_datadir", help="Path to the in-distribution data folder.")
    # parser.add_argument("--out_datadir", help="Path to the out-of-distribution data folder.")
    
    parser.add_argument('--ood_method', choices=OOD_METHOD_CHOICES, required=True, help='OOD detection method to use')

    # arguments for ODIN
    parser.add_argument('--temperature_odin', default=1000, type=int,
                        help='temperature scaling for odin')
    parser.add_argument('--epsilon_odin', default=0.0, type=float,
                        help='perturbation magnitude for odin')

    # arguments for Energy and ODIN
    parser.add_argument('--temperature', default=1, type=int,
                        help='temperature scaling for energy')
    """
    # arguments for Mahalanobis
    parser.add_argument('--mahalanobis_param_path', default='checkpoints/finetune/tune_mahalanobis',
                        help='path to tuned mahalanobis parameters')

    # arguments for GradNorm
    parser.add_argument('--temperature_gradnorm', default=1, type=float,
                        help='temperature scaling for GradNorm')
    # arguments for React
    parser.add_argument('--temperature_react', default=1, type=float,
                        help='temperature scaling for React')
    # arguments for RankFeat
    parser.add_argument('--temperature_rankfeat', default=1, type=float,
                        help='temperature scaling for RankFeat')
    """
    print('******************************************')
    main(parser.parse_args())