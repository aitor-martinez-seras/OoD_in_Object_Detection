from pathlib import Path
from datetime import datetime
import argparse
import json
from typing import Literal, List
import os

from tap import Tap
import torch
from torchvision.ops import roi_align
from torchvision.utils import draw_bounding_boxes

from umap import UMAP
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from ultralytics import YOLO
from ultralytics.yolo.utils.callbacks import tensorboard
from ultralytics.yolo.utils import yaml_load
from ultralytics.yolo.utils import DEFAULT_CFG_PATH

from data_utils import read_json, write_json, load_dataset_and_dataloader, create_TAO_dataset_and_dataloader
from ood_utils import ActivationsExtractor, configure_extra_output_of_the_model, OODMethod

# Constants
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.15


class SimpleArgumentParser(Tap):
    # Required arguments
    model_folder: str  # Which model to use
    device: int  # Device to use for training on GPU. Indicate more than one to use multiple GPUs. Use -1 for CPU.
    number_of_known_classes: int  # Number of known classes
    one_umap_per_stride: bool  # If True, one UMAP per stride. If False, all strides in one UMAP
    mode: Literal['umap', 'pca_umap', 'pca']  # Mode to use for the UMAP representation
    # Optional arguments
    n_neighbors: int = 20  # Number of neighbors to use for UMAP
    metric: Literal["euclidean", "manhattan", "cosine"] = "cosine"  # Metric to use for UMAP
    min_dist: float = 0.01  # Minimum distance to use for UMAP
    target_weight: float = 0.5  # Target weight to use for UMAP
    grid_search_umap: bool = False  # If True, grid search for UMAP parameters
    dataset: str = "tao_coco"
    split: Literal["train", "val", "test"] = "val"  # Which split of the dataset to use
    batch_size: int = 16  # Batch size.
    workers: int = 8  # Number of background threads used to load data.

    def __init__(self, *args, **kwargs):
        super().__init__(explicit_bool=True, *args, **kwargs)

    def configure(self):
        self.add_argument("-d", "--device")
        self.add_argument("-nkc", "--number_of_known_classes")
        #self.add_argument("--grid_search_umap", action="store_true")


def create_and_plot_one_stride(known_activations_one_stride, known_labels_one_stride, unknown_activations_one_stride, unknown_labels_one_stride,
                               CLASSES, known_classes, unknown_classes, 
                               n_neighbors, metric, min_dist, target_weight, save_folder,
                               mode, stride):
        params = {
            'n_neighbors': n_neighbors,
            'metric': metric,
            'min_dist': min_dist,
            'target_weight': target_weight
        }
        print(f" -- Params: {params} --")
        # Create a new UMAP instance with the current set of parameters
        print(f"Fitting representations...")
        if mode == 'umap':
            my_umap = UMAP(n_components=2, n_neighbors=n_neighbors, metric=metric, min_dist=min_dist, target_weight=target_weight)
            embedding = my_umap.fit_transform(known_activations_one_stride, y=known_labels_one_stride)
        elif mode == 'pca_umap':
            pca = PCA(n_components=50)
            my_umap = UMAP(n_components=2, n_neighbors=n_neighbors, metric=metric, min_dist=min_dist, target_weight=target_weight)
            filtered_known_activations_pca = pca.fit_transform(known_activations_one_stride)
            embedding = my_umap.fit_transform(filtered_known_activations_pca, y=known_labels_one_stride)
        elif mode == 'pca':
            pca = PCA(n_components=2)
            embedding = pca.fit_transform(known_activations_one_stride)
        print(f"Representations fitted!")

        # Transform the unknown classes data
        print(f"Transforming UMAP unknown representations...")
        if mode == 'umap':
            embedding_unknown = my_umap.transform(unknown_activations_one_stride)
        elif mode == 'pca_umap':
            unknown_activations_pca = pca.transform(unknown_activations_one_stride)
            embedding_unknown = my_umap.transform(unknown_activations_pca)
        elif mode == 'pca':
            embedding_unknown = pca.transform(unknown_activations_one_stride)
        print(f"Unknown representations transformed!")

        print(f"Saving KNOWN representation...")
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(1, 1, 1)#, projection='3d')
        cmap = plt.cm.tab20(np.arange(40).astype(int))
        # Represent the known classes
        color_idx = 0  # For both known and unknown classes, to distinguish them
        for idx_cls, cl in enumerate(known_classes):
            plotEmbeddings = np.array([embedding[i,:] for i in range(len(embedding)) if known_labels_one_stride[i]==cl])
            if len(plotEmbeddings)!=0:
                ax.scatter(*plotEmbeddings.T, color=cmap[color_idx],label=CLASSES[cl],alpha=0.7)
                color_idx = color_idx + 1
        
        plt.legend()
        # Save fig with parameters in the name
        fig.savefig(save_folder / f"{mode}_s{stride}_{params}_known.png")

        # Represent the unknown classes
        print(f"Saving UNKNOWN representation...")
        for idx_cls, cl in enumerate(unknown_classes):
            if idx_cls >= 15:
                break
            plotEmbeddings_unk = np.array([embedding_unknown[i,:] for i in range(len(embedding_unknown)) if unknown_labels_one_stride[i]==cl])
            if len(plotEmbeddings_unk) > 50:
                ax.scatter(*plotEmbeddings_unk.T, color=cmap[color_idx], label=CLASSES[cl], alpha=0.7, marker='s')
                color_idx = color_idx + 1
        
        plt.legend()
        # Save fig with parameters in the name
        fig.savefig(save_folder / f"{mode}_s{stride}_{params}_unknown.png")
        plt.close()
        print(f"Representations saved for mode {mode} with params: {params}!")


def main():

    # Constants
    ROOT = Path().cwd()  # Assumes this script is in the root of the project
    FIGS_PATH = ROOT / 'figures'
    NOW = datetime.now().strftime("%Y%m%d_%H%M")
    # Output sizes for all yolo models for strides 8, 16 and 32 respectively
    OUTPUT_SIZES = {
        'n': ((10, 10), (10, 5), (5, 5)),
        's': ((20, 20), (10, 10), (5, 5)),
        'm': ((40, 40), (20, 20), (10, 10)),
        'l': ((80, 80), (40, 40), (20, 20)),
        'x': ((160, 160), (80, 80), (40, 40)),
    }
    o_s = 7
    OUTPUT_SIZES = [(o_s, o_s), (o_s, o_s), (o_s, o_s)]
    
    # Load model
    print('Loading model...')

    ######## IMPORTANT NOTE: ###########
    # To work with CUDA and predictions, we must define the CUDA_VISIBLE_DEVICES environment variable
    # as Ultralytics automatically does it internally and therefore then creates a big mess of GPUs
    device = f"cuda:{args.device}" if args.device != -1 else "cpu"
    if device != "cpu":
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        device = 'cuda:0'

    # Load model and set the mode
    model_folder_path = ROOT / args.model_folder
    model = YOLO(model=model_folder_path / 'weights' / 'best.pt')
    # Modify internal attributes of the model to obtain the desired outputs in the extra_item
    model.model.extraction_mode = 'all_ftmaps'
    print('Model loaded!')
    
    # Dataset selection
    yaml_file = f"{args.dataset}.yaml"
    # if 'tao' in args.dataset:
    #     dataset, dataloader = create_TAO_dataset_and_dataloader(
    #             yaml_file,
    #             args=args,
    #             data_split=args.split,
    #             batch_size=args.batch_size,
    #             workers=args.workers,
    #     )
    # elif 'coco' in args.dataset:
    dataset, dataloader = load_dataset_and_dataloader(
            dataset_name=args.dataset,
            data_split=args.split,
            batch_size=args.batch_size,
            workers=args.workers,
            owod_task=''
        )
    # else:
    #     raise ValueError(f"Dataset {args.dataset} not supported")
    CLASSES = dataset.data["names"]

    # Define the folder path
    folder_for_model_in_figs = FIGS_PATH / f'{model_folder_path.name}'
    folder_for_model_in_figs.mkdir(exist_ok=True)
    folder_to_save_figs = folder_for_model_in_figs / f'{args.dataset}_{args.split}_{args.number_of_known_classes}_classes'
    folder_to_save_figs.mkdir(exist_ok=True)

    # Process data
    print('Extracting activations...')

    # List of features per stride
    all_feature_maps_per_stride = [[] for _ in range(3)]
    all_cls_labels = []

    # Start iterating over the data
    number_of_batches = len(dataloader)
    for idx_of_batch, data in enumerate(dataloader):
        
        if idx_of_batch % 50 == 0:
            print(f"{(idx_of_batch/number_of_batches) * 100:02.1f}%: Procesing batch {idx_of_batch} of {number_of_batches}")
            
        ### Prepare images and targets to feed the model ###
        imgs = data['img'].to(device)
        targets = OODMethod.create_targets_dict(data)

        ### Process the images to get the results (bboxes and clasification) and the extra info for OOD detection ###
        results = model.predict(imgs, save=False, verbose=False, device=device)

        ### Extract the RoI Aligned features for each target ###
        # Create a list with 3 levels and each level with a tensor of shape [N, C, H, W]
        ftmaps_per_level = []
        for idx_lvl in range(3):
            one_level_ftmaps = []
            for i_bt in range(len(results)):
                one_level_ftmaps.append(results[i_bt].extra_item[idx_lvl].to('cpu'))
            ftmaps_per_level.append(torch.stack(one_level_ftmaps))
            
        for idx_lvl in range(3):
            if args.one_umap_per_stride:
                output_size = OUTPUT_SIZES[2]
            else:
                output_size = OUTPUT_SIZES[idx_lvl]
            
            # For each stride or lvl, different RoIAlign parameters
            _bboxes = [b.to(torch.float32) for b in targets["bboxes"]]
            roi_aligned_ftmaps_one_stride = roi_align(
                input=ftmaps_per_level[idx_lvl],
                boxes=_bboxes,
                output_size=output_size,  # Output sizes for same size of features in YOLOv8n: s8(10,10), s16(10,5), s32(5,5)
                spatial_scale=ftmaps_per_level[idx_lvl].shape[2]/imgs[0].shape[2]
            )

            # roi_aligned_ftmaps_one_stride = gap_roi_align(
            #     input=ftmaps_per_level[idx_lvl],
            #     boxes=_bboxes,
            #     output_size=output_size,  # Output sizes for same size of features in YOLOv8n: s8(10,10), s16(10,5), s32(5,5)
            #     spatial_scale=ftmaps_per_level[idx_lvl].shape[2]/imgs[0].shape[2]
            # )

            all_feature_maps_per_stride[idx_lvl].append(roi_aligned_ftmaps_one_stride)
        # Keep track of the labels
        all_cls_labels.append(torch.hstack(targets['cls']))

        # Represent some images to check if the data is correct
        if idx_of_batch in [0, 150, 805]:
            for idx_img in range(5,8):
                img = imgs[idx_img].cpu()
                # Bounding boxes
                img = draw_bounding_boxes(
                    image=img,
                    boxes=targets['bboxes'][idx_img].cpu(),
                    labels=[CLASSES[int(lbl)] for lbl in targets['cls'][idx_img].cpu().numpy()],
                )
                plt.imshow(img.permute(1, 2, 0).numpy())
                plt.savefig(folder_to_save_figs / f'example_imgs_batch_{idx_of_batch}_img_{idx_img}.png')
                plt.close()

    # To tensors
    for i in range(len(all_feature_maps_per_stride)):
        all_feature_maps_per_stride[i] = torch.cat(all_feature_maps_per_stride[i], dim=0)
    all_cls_labels = torch.hstack(all_cls_labels)
    print('Activations extracted!')

    ### Create the UMAP representation ###
    print('*** Creating UMAP representation ***')

    # Define the known and unknown classes
    known_classes = np.arange(args.number_of_known_classes)
    unknown_classes = np.array([c for c in range(len(CLASSES)) if c not in known_classes])
    # Convert known and unknown class indices to tensors for efficient comparison
    known_classes_tensor = torch.tensor(known_classes, dtype=torch.long)
    unknown_classes_tensor = torch.tensor([c for c in range(len(CLASSES)) if c not in known_classes], dtype=torch.long)
    # Positions of known and unknown classes can be calculated more efficiently
    positions_of_known_classes = torch.isin(all_cls_labels, known_classes_tensor)
    positions_of_unknown_classes = torch.isin(all_cls_labels, unknown_classes_tensor)
    print(f"Number of classes: {len(CLASSES)}")
    print(f"Known classes: {[CLASSES[cl] for cl in known_classes]}\n")
    print(f"Unknown classes: {[CLASSES[cl] for cl in unknown_classes]}\n")
    
    """
    # Filter the classes. Flaten the activations and convert them into arrays to use UMAP
    filtered_known_activations_per_stride = [one_stride[positions_of_known_classes].flatten(start_dim=1).numpy() for one_stride in all_feature_maps_per_stride]
    filtered_known_labels = all_cls_labels[positions_of_known_classes].numpy()
    filtered_unknown_activations_per_stride = [one_stride[positions_of_unknown_classes].flatten(start_dim=1).numpy() for one_stride in all_feature_maps_per_stride]
    filtered_unknown_labels = all_cls_labels[positions_of_unknown_classes].numpy()
    """

    print('*** Creating UMAP representation ***')  # https://umap-learn.readthedocs.io/en/latest/parameters.html 
    if args.one_umap_per_stride:

        filtered_known_activations_per_stride = []
        filtered_unknown_activations_per_stride = []
        for one_stride_activations in all_feature_maps_per_stride:
            # Flatten the activations for UMAP. This step can be combined with filtering to avoid redundant operations
            flattened_activations = one_stride_activations.flatten(start_dim=1)
            
            # Filter and convert to numpy in one step
            filtered_known_activations = flattened_activations[positions_of_known_classes].numpy()
            filtered_unknown_activations = flattened_activations[positions_of_unknown_classes].numpy()
            
            filtered_known_activations_per_stride.append(filtered_known_activations)
            filtered_unknown_activations_per_stride.append(filtered_unknown_activations)

        # Since labels for known and unknown classes don't change across strides, you can filter them once outside the loop
        filtered_known_labels = all_cls_labels[positions_of_known_classes].numpy()
        filtered_unknown_labels = all_cls_labels[positions_of_unknown_classes].numpy()

        # Further filter the unknown classes to extract only the 15 with the most samples
        # This is to avoid having too many samples of unknown classes, which could make the UMAP representation less clear


        # For all strides
        for idx_stride in range(3):
            print(f'- Stride {(idx_stride+1)*8} -')
            
            # Grid search
            if args.grid_search_umap:
                
                grid_search_folder = folder_for_model_in_figs / f'grid_search_umap_{model_folder_path.name}'
                grid_search_folder.mkdir(exist_ok=True)
                grid_search_folder_for_dataset = grid_search_folder / f'{args.dataset}'
                grid_search_folder_for_dataset.mkdir(exist_ok=True)
                grid_search_folder_one_stride = grid_search_folder_for_dataset / f'stride_{(idx_stride+1)*8}'
                grid_search_folder_one_stride.mkdir(exist_ok=True)

                from itertools import product
                umap_args_to_test = {
                    # "n_neighbors": [5, 20, 100],
                    # "min_dist": [0.01, 0.1, 0.5],
                    # "target_weight": [0.2, 0.5, 0.8],
                    # "metric": ["euclidean", "cosine"]
                    "n_neighbors": [20],
                    "min_dist": [0.01, 0.1],
                    "target_weight": [0.2, 0.5, 0.8],
                    "metric": ["manhattan", "euclidean", "cosine"]
                }
                # Generate all combinations of parameters
                param_combinations = list(product(*umap_args_to_test.values()))
                for combination in param_combinations:
                    # Map the parameter combination to the corresponding keyword arguments
                    params = dict(zip(umap_args_to_test.keys(), combination))

                    create_and_plot_one_stride(
                        known_activations_one_stride=filtered_known_activations_per_stride[idx_stride],
                        known_labels_one_stride=filtered_known_labels,
                        unknown_activations_one_stride=filtered_unknown_activations_per_stride[idx_stride],
                        unknown_labels_one_stride=filtered_unknown_labels,
                        CLASSES=CLASSES,
                        known_classes=known_classes,
                        unknown_classes=unknown_classes,
                        n_neighbors=params['n_neighbors'],
                        metric=params['metric'],
                        min_dist=params['min_dist'],
                        target_weight=params['target_weight'],
                        save_folder=grid_search_folder_one_stride,
                        mode=args.mode,
                        stride=(idx_stride+1)*8
                    )

                        
                    # print(f" -- Params: {params} --")
                    # # Create a new UMAP instance with the current set of parameters
                    # my_umap = UMAP(n_components=2, **params)
                    # embedding = my_umap.fit_transform(filtered_known_activations_per_stride[idx_stride], y=filtered_known_labels)
                    # print(f"UMAP representation fitted!")

                    # # Transform the unknown classes data
                    # print(f"Transforming UMAP unknown representations...")
                    # embedding_unknown = my_umap.transform(filtered_unknown_activations_per_stride[idx_stride])
                    # print(f"UMAP unknown representations transformed!")

                    # print(f"Saving UMAP representation...")
                    # fig = plt.figure(figsize=(14, 10))
                    # ax = fig.add_subplot(1, 1, 1)#, projection='3d')
                    
                    # cmap = plt.cm.tab20(np.arange(40).astype(int))
                    # # Represent the known classes
                    # color_idx = 0  # For both known and unknown classes, to distinguish them
                    # for idx_cls, cl in enumerate(known_classes):
                    #     plotEmbeddings = np.array([embedding[i,:] for i in range(len(embedding)) if filtered_known_labels[i]==cl])
                    #     if len(plotEmbeddings)!=0:
                    #         ax.scatter(*plotEmbeddings.T, color=cmap[color_idx],label=CLASSES[cl],alpha=0.7)
                    #         color_idx = color_idx + 1
                    
                    # plt.legend()
                    # # Save fig with parameters in the name
                    # fig.savefig(grid_search_folder_one_stride / f"umap_params_{params}_known.png")

                    # # Represent the unknown classes
                    # for idx_cls, cl in enumerate(unknown_classes):
                    #     if idx_cls >= 15:
                    #         break
                    #     plotEmbeddings_unk = np.array([embedding_unknown[i,:] for i in range(len(embedding_unknown)) if filtered_unknown_labels[i]==cl])
                    #     if len(plotEmbeddings_unk) > 50:
                    #         ax.scatter(*plotEmbeddings_unk.T, color=cmap[color_idx], label=CLASSES[cl], alpha=0.7, marker='s')
                    #         color_idx = color_idx + 1
                    
                    # plt.legend()
                    # # Save fig with parameters in the name
                    # fig.savefig(grid_search_folder_one_stride / f"umap_params_{params}_unknown.png")
                    # plt.close()
                    # print(f"UMAP saved for params: {params}!")
                
            # Normal execution, just one UMAP
            else:

                create_and_plot_one_stride(
                        known_activations_one_stride=filtered_known_activations_per_stride[idx_stride],
                        known_labels_one_stride=filtered_known_labels,
                        unknown_activations_one_stride=filtered_unknown_activations_per_stride[idx_stride],
                        unknown_labels_one_stride=filtered_unknown_labels,
                        CLASSES=CLASSES,
                        known_classes=known_classes,
                        unknown_classes=unknown_classes,
                        n_neighbors=args.n_neighbors,
                        metric=args.metric,
                        min_dist=args.min_dist,
                        target_weight=args.target_weight,
                        save_folder=folder_to_save_figs,
                        mode=args.mode,
                        stride=(idx_stride+1)*8
                    )

                # print(f"Fitting UMAP representation...")
                # # First reduce the input to 50 dimensions with PCA
                # from sklearn.decomposition import PCA
                # pca = PCA(n_components=50)
                # filtered_known_activations_pca = pca.fit_transform(filtered_known_activations_per_stride[idx_stride])

                # # Create the UMAP object and fit the known classes data
                # my_umap = UMAP(n_components=2, n_neighbors=20, metric='cosine', min_dist=0.01, target_weight=1)
                # embedding = my_umap.fit_transform(filtered_known_activations_pca, y=filtered_known_labels)
                # #embedding = my_umap.fit_transform(filtered_known_activations_per_stride[idx_stride], y=filtered_known_labels)
                # print(f"UMAP representation fitted!")

                # # Transform the unknown classes data
                # print(f"Transforming UMAP unknown representations...")
                # embedding_unknown = my_umap.transform(pca.transform(filtered_unknown_activations_per_stride[idx_stride]))
                # #embedding_unknown = my_umap.transform(filtered_unknown_activations_per_stride[idx_stride])
                # print(f"UMAP unknown representations transformed!")

                # print(f"Saving UMAP representation...")
                # fig = plt.figure(figsize=(14, 10))
                # ax = fig.add_subplot(1, 1, 1)#, projection='3d')
                
                # cmap = plt.cm.tab20(np.arange(20).astype(int))
                # # Represent the known classes
                # color_idx = 0  # For both known and unknown classes, to distinguish them
                # for idx_cls, cl in enumerate(known_classes):
                #     plotEmbeddings = np.array([embedding[i,:] for i in range(len(embedding)) if filtered_known_labels[i]==cl])
                #     if len(plotEmbeddings)!=0:
                #         ax.scatter(*plotEmbeddings.T, color=cmap[color_idx],label=CLASSES[cl],alpha=0.7)
                #         color_idx = color_idx + 1

                # plt.legend()
                # fig.savefig(folder_to_save_figs /f"umap_known_stride_{(idx_stride+1)*8}.png")

                # # Represent the unknown classes
                # for idx_cls, cl in enumerate(unknown_classes):
                #     if idx_cls >= 15:
                #         break
                #     plotEmbeddings_unk = np.array([embedding_unknown[i,:] for i in range(len(embedding_unknown)) if filtered_unknown_labels[i]==cl])
                #     if len(plotEmbeddings_unk) > 50:
                #         ax.scatter(*plotEmbeddings_unk.T, color=cmap[color_idx], label=CLASSES[cl], alpha=0.7, marker='s')
                #         color_idx = color_idx + 1
                #     # ax.scatter(*plotEmbeddings_unk.T, color=cmap[color_idx], label=CLASSES[cl], alpha=0.7, marker='s')
                #     # color_idx = color_idx + 1
                #     # if len(plotEmbeddings_unk) > 200:
                #         # ax.scatter(*plotEmbeddings_unk.T, color=cmap[color_idx], label=CLASSES[cl], alpha=0.7, marker='s')
                #         # color_idx = color_idx + 1
                
                # plt.legend()
                # fig.savefig(folder_to_save_figs / f"umap_unknown_stride_{(idx_stride+1)*8}.png")
                # plt.close()
                # print(f"UMAP representation saved!")


    else:  # All strides in one UMAP, with different markers each stride
        # Therefore, we have to make a big array with all the activations of all strides
        # We also have to have a big array with the labels of all strides, by repeating the array of labels 3 times
        raise NotImplementedError("Not implemented yet")


if __name__ == "__main__":
    args = SimpleArgumentParser().parse_args()
    print(args)
    main()


###############################
    # # FOR FIRST STRIDE

    # # Create the UMAP object and fit the known classes data
    # my_umap = UMAP(n_components=2,n_neighbors=20, metric='cosine',min_dist=0.01,target_weight=0.8)
    # embedding = my_umap.fit_transform(filtered_known_activations_per_stride[0], y=filtered_known_labels)

    # # Transform the unknown classes data
    # embedding_unknown = my_umap.transform(filtered_unknown_activations_per_stride[0])

    # fig = plt.figure(figsize=(14, 10))
    # ax = fig.add_subplot(1, 1, 1)#, projection='3d')
    
    # cmap = plt.cm.tab20(np.arange(40).astype(int))
    # # Represent the known classes
    # color_idx = 0
    # for idx_cls, cl in enumerate(known_classes):
    #     plotEmbeddings = np.array([embedding[i,:] for i in range(len(embedding)) if filtered_known_labels[i]==cl])
    #     if len(plotEmbeddings)!=0:
    #         ax.scatter(*plotEmbeddings.T, color=cmap[color_idx],label=CLASSES[cl],alpha=0.7)
    #         color_idx = color_idx + 1

    # # Represent the unknown classes
    # for idx_cls, cl in enumerate(unknown_classes):
    #     plotEmbeddings_unk = np.array([embedding_unknown[i,:] for i in range(len(embedding_unknown)) if filtered_unknown_labels[i]==cl])
    #     if len(plotEmbeddings_unk) > 200:
    #         ax.scatter(*plotEmbeddings_unk.T, color=cmap[color_idx], label=CLASSES[cl],alpha=0.7, marker='s')
    #         color_idx = color_idx + 1
    
    # plt.legend()
    # fig.savefig(f"umap.png")
    # plt.close()

    ### GAP RoI Align ###
def _bilinear_interpolate(
    input,  # [N, C, H, W]
    roi_batch_ind,  # [K]
    y,  # [K, PH, IY]
    x,  # [K, PW, IX]
    ymask,  # [K, IY]
    xmask,  # [K, IX]
):
    _, channels, height, width = input.size()

    # deal with inverse element out of feature map boundary
    y = y.clamp(min=0)
    x = x.clamp(min=0)
    y_low = y.int()
    x_low = x.int()
    y_high = torch.where(y_low >= height - 1, height - 1, y_low + 1)
    y_low = torch.where(y_low >= height - 1, height - 1, y_low)
    y = torch.where(y_low >= height - 1, y.to(input.dtype), y)

    x_high = torch.where(x_low >= width - 1, width - 1, x_low + 1)
    x_low = torch.where(x_low >= width - 1, width - 1, x_low)
    x = torch.where(x_low >= width - 1, x.to(input.dtype), x)

    ly = y - y_low
    lx = x - x_low
    hy = 1.0 - ly
    hx = 1.0 - lx

    # do bilinear interpolation, but respect the masking!
    # TODO: It's possible the masking here is unnecessary if y and
    # x were clamped appropriately; hard to tell
    def masked_index(
        y,  # [K, PH, IY]
        x,  # [K, PW, IX]
    ):
        if ymask is not None:
            assert xmask is not None
            y = torch.where(ymask[:, None, :], y, 0)
            x = torch.where(xmask[:, None, :], x, 0)
        return input[
            roi_batch_ind[:, None, None, None, None, None],
            torch.arange(channels, device=input.device)[None, :, None, None, None, None],
            y[:, None, :, None, :, None],  # prev [K, PH, IY]
            x[:, None, None, :, None, :],  # prev [K, PW, IX]
        ]  # [K, C, PH, PW, IY, IX]

    v1 = masked_index(y_low, x_low)
    v2 = masked_index(y_low, x_high)
    v3 = masked_index(y_high, x_low)
    v4 = masked_index(y_high, x_high)

    # all ws preemptively [K, C, PH, PW, IY, IX]
    def outer_prod(y, x):
        return y[:, None, :, None, :, None] * x[:, None, None, :, None, :]

    w1 = outer_prod(hy, hx)
    w2 = outer_prod(hy, lx)
    w3 = outer_prod(ly, hx)
    w4 = outer_prod(ly, lx)

    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    return val


# TODO: this doesn't actually cache
# TODO: main library should make this easier to do
def maybe_cast(tensor):
    if torch.is_autocast_enabled() and tensor.is_cuda and tensor.dtype != torch.double:
        return tensor.float()
    else:
        return tensor

import collections
from itertools import repeat
def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse

_pair = _ntuple(2, "_pair")



# def _gap_roi_align(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned):
#     orig_dtype = input.dtype

#     input = maybe_cast(input)
#     rois = maybe_cast(rois)

#     _, _, height, width = input.size()

#     # input: [N, C, H, W]
#     # rois: [K, 5]

#     roi_batch_ind = rois[:, 0].int()  # [K]
#     offset = 0.5 if aligned else 0.0
#     roi_start_w = rois[:, 1] * spatial_scale - offset  # [K]
#     roi_start_h = rois[:, 2] * spatial_scale - offset  # [K]
#     roi_end_w = rois[:, 3] * spatial_scale - offset  # [K]
#     roi_end_h = rois[:, 4] * spatial_scale - offset  # [K]

#     roi_width = roi_end_w - roi_start_w  # [K]
#     roi_height = roi_end_h - roi_start_h  # [K]
#     if not aligned:
#         roi_width = torch.clamp(roi_width, min=1.0)  # [K]
#         roi_height = torch.clamp(roi_height, min=1.0)  # [K]

#     exact_sampling = sampling_ratio > 0

#     roi_bin_grid_h = sampling_ratio if exact_sampling else torch.ceil(roi_height / pooled_height)  # scalar or [K]
#     roi_bin_grid_w = sampling_ratio if exact_sampling else torch.ceil(roi_width / pooled_width)  # scalar or [K]

#     if exact_sampling:
#         iy = torch.arange(roi_bin_grid_h, device=input.device)  # [IY]
#         ix = torch.arange(roi_bin_grid_w, device=input.device)  # [IX]
#     else:
#         iy = torch.arange(height, device=input.device)  # [IY]
#         ix = torch.arange(width, device=input.device)  # [IX]

#     def from_K(t):
#         return t[:, None, None]

#     y = (
#         from_K(roi_start_h)
#         + (iy[None, None, :] + 0.5).to(input.dtype) * from_K(roi_height / roi_bin_grid_h)
#     )  # [K, 1, IY]
#     x = (
#         from_K(roi_start_w)
#         + (ix[None, None, :] + 0.5).to(input.dtype) * from_K(roi_width / roi_bin_grid_w)
#     )  # [K, 1, IX]

#     # Perform bilinear interpolation for the values at (y, x) locations
#     val = _bilinear_interpolate(input, roi_batch_ind, y, x)  # [K, C, 1, 1, IY, IX]

#     # Perform Global Average Pooling over the RoI
#     output = val.mean((-1, -2))  # Take the mean across spatial dimensions IY, IX

#     output = output.to(orig_dtype)  # Convert back to the original dtype if needed

#     return output.squeeze(-1).squeeze(-1)


def _roi_align_global_avg_pooling(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned):
    orig_dtype = input.dtype

    input = maybe_cast(input)
    rois = maybe_cast(rois)

    _, _, height, width = input.size()

    # input: [N, C, H, W]
    # rois: [K, 5]

    roi_batch_ind = rois[:, 0].int()  # [K]
    offset = 0.5 if aligned else 0.0
    roi_start_w = rois[:, 1] * spatial_scale - offset  # [K]
    roi_start_h = rois[:, 2] * spatial_scale - offset  # [K]
    roi_end_w = rois[:, 3] * spatial_scale - offset  # [K]
    roi_end_h = rois[:, 4] * spatial_scale - offset  # [K]

    roi_width = roi_end_w - roi_start_w  # [K]
    roi_height = roi_end_h - roi_start_h  # [K]
    if not aligned:
        roi_width = torch.clamp(roi_width, min=1.0)  # [K]
        roi_height = torch.clamp(roi_height, min=1.0)  # [K]

    # Calculate the total number of samples per ROI for averaging
    exact_sampling = sampling_ratio > 0
    if exact_sampling:
        total_samples_per_roi = sampling_ratio * sampling_ratio
    else:
        total_samples_per_roi = (torch.ceil(roi_height / pooled_height) * torch.ceil(roi_width / pooled_width)).to(input.dtype)

    def from_K(t):
        return t[:, None, None]

    y = (
        from_K(roi_start_h)
        + (torch.arange(height, device=input.device)[None, None, :] + 0.5).to(input.dtype) * from_K(roi_height / height)
    )  # Adjusted for global pooling: [K, 1, IY]
    x = (
        from_K(roi_start_w)
        + (torch.arange(width, device=input.device)[None, None, :] + 0.5).to(input.dtype) * from_K(roi_width / width)
    )  # Adjusted for global pooling: [K, 1, IX]
    
    # Perform bilinear interpolation
    val = _bilinear_interpolate(input, roi_batch_ind, y, x, None, None)  # [K, C, 1, 1, IY, IX]
    
    # Sum over the entire RoI and divide by the total number of samples for global average pooling
    output = val.sum((-1, -2)) / total_samples_per_roi[:, None, None, None]  # [K, C]

    output = output.to(orig_dtype)  # Convert back to the original dtype

    return output


from torchvision.ops._utils import convert_boxes_to_roi_format, check_roi_boxes_shape
def gap_roi_align(input, boxes, output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False):

    check_roi_boxes_shape(boxes)
    rois = boxes
    output_size = _pair(output_size)
    if not isinstance(rois, torch.Tensor):
        rois = convert_boxes_to_roi_format(rois)
    return _roi_align_global_avg_pooling(input, rois, spatial_scale, output_size[0], output_size[1], sampling_ratio, aligned)
