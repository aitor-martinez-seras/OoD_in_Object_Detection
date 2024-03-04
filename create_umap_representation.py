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

from ultralytics import YOLO
from ultralytics.yolo.utils.callbacks import tensorboard
from ultralytics.yolo.utils import yaml_load
from ultralytics.yolo.utils import DEFAULT_CFG_PATH

from data_utils import read_json, write_json, create_YOLO_dataset_and_dataloader, build_dataloader, create_TAO_dataset_and_dataloader
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
    # Optional arguments
    dataset: str = "tao_coco"
    batch_size: int = 16  # Batch size.
    workers: int = 8  # Number of background threads used to load data.

    def __init__(self, *args, **kwargs):
        super().__init__(explicit_bool=True, *args, **kwargs)

    def configure(self):
        self.add_argument("-d", "--device")
        self.add_argument("-nkc", "--number_of_known_classes")


def main():

    # Constants
    ROOT = Path().cwd()  # Assumes this script is in the root of the project
    FIGS_PATH = ROOT / 'figures'
    NOW = datetime.now().strftime("%Y%m%d_%H%M")
    OUTPUT_SIZES = [(10, 10), (10, 5), (5, 5)]  # Output sizes for stride 8, 16 and 32 respectively
    
    # Load model
    print('Loading model...')

    ######## IMPORTANT NOTE: ###########
    # To work with CUDA and predictions, we must define the CUDA_VISIBLE_DEVICES environment variable
    # as Ultralytics automatically does it internally and therefore then creates a big mess of GPUs
    device = f"cuda:{args.device}" if args.device != -1 else "cpu"
    if device != "cpu":
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        device = 'cuda:0'
    model_folder_path = ROOT / args.model_folder
    model = YOLO(model=model_folder_path / 'weights' / 'best.pt')
    # Modify internal attributes of the model to obtain the desired outputs in the extra_item
    model.model.modo = 'all_ftmaps'
    print('Model loaded!')
    
    # Dataset selection
    yaml_file = f"{args.dataset}.yaml"
    dataset, dataloader = create_TAO_dataset_and_dataloader(
            yaml_file,
            args,
            data_split='train',
    )
    CLASSES = dataset.data["names"]

    # Process data
    print('Extracting activations...')

    # List of features per stride
    all_feature_maps_per_stride = [[] for _ in range(3)]
    all_cls_labels = []

    # Obtain the bbox format from the last transform of the dataset
    if hasattr(dataloader.dataset.transforms.transforms[-1], "bbox_format"):
        bbox_format = dataloader.dataset.transforms.transforms[-1].bbox_format
    else:
        bbox_format=dataloader.dataset.labels[0]['bbox_format']

    # Start iterating over the data
    number_of_batches = len(dataloader)
    for idx_of_batch, data in enumerate(dataloader):
        
        if idx_of_batch % 50 == 0:
            print(f"{(idx_of_batch/number_of_batches) * 100:02.1f}%: Procesing batch {idx_of_batch} of {number_of_batches}")
            
        ### Prepare images and targets to feed the model ###
        imgs = data['img'].to(device)
        targets = OODMethod.create_targets_dict(data, bbox_format)

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
            output_size = OUTPUT_SIZES[idx_lvl] if args.one_umap_per_stride else OUTPUT_SIZES[2]
            # For each stride or lvl, different RoIAlign parameters
            _bboxes = [b.to(torch.float32) for b in targets["bboxes"]]
            roi_aligned_ftmaps_one_stride = roi_align(
                input=ftmaps_per_level[idx_lvl],
                boxes=_bboxes,
                output_size=(10, 10),  # Output sizes for same size of features in YOLOv8n: s8(10,10), s16(10,5), s32(5,5)
                spatial_scale=ftmaps_per_level[idx_lvl].shape[2]/imgs[0].shape[2]
            )

            all_feature_maps_per_stride[idx_lvl].append(roi_aligned_ftmaps_one_stride)
        # Keep track of the labels
        all_cls_labels.append(torch.hstack(targets['cls']))

        # Represent some images to check if the data is correct
        if idx_of_batch in [0, 150, 333, 805]:
            for idx_img in range(5,8):
                img = imgs[idx_img].cpu()
                # Bounding boxes
                img = draw_bounding_boxes(
                    image=img,
                    boxes=targets['bboxes'][idx_img].cpu(),
                    labels=[CLASSES[int(lbl)] for lbl in targets['cls'][idx_img].cpu().numpy()],
                )
                plt.imshow(img.permute(1, 2, 0).numpy())
                plt.savefig(FIGS_PATH / f'example_imgs_{model_folder_path.name}_batch_{idx_of_batch}_img_{idx_img}.png')
                plt.close()

    # To tensors
    for i in range(len(all_feature_maps_per_stride)):
        all_feature_maps_per_stride[i] = torch.cat(all_feature_maps_per_stride[i], dim=0)
    all_cls_labels = torch.hstack(all_cls_labels)
    print('Activations extracted!')

    ### Create the UMAP representation ###

    # Define the known and unknown classes
    known_classes = np.arange(args.number_of_known_classes)
    unknown_classes = np.array([c for c in range(len(CLASSES)) if c not in known_classes])
    positions_of_known_classes = torch.isin(all_cls_labels, torch.from_numpy(known_classes))
    positions_of_unknown_classes = torch.isin(all_cls_labels, torch.from_numpy(unknown_classes))
    # Filter the classes. Flaten the activations and convert them into arrays to use UMAP
    filtered_known_activations_per_stride = [one_stride[positions_of_known_classes].flatten(start_dim=1).numpy() for one_stride in all_feature_maps_per_stride]
    filtered_known_labels = all_cls_labels[positions_of_known_classes].numpy()
    filtered_unknown_activations_per_stride = [one_stride[positions_of_unknown_classes].flatten(start_dim=1).numpy() for one_stride in all_feature_maps_per_stride]
    filtered_unknown_labels = all_cls_labels[positions_of_unknown_classes].numpy()
    
    if args.one_umap_per_stride:    

        # For all strides
        for idx_stride in range(3):

            # Create the UMAP object and fit the known classes data
            my_umap = UMAP(n_components=2, n_neighbors=20, metric='cosine', min_dist=0.01, target_weight=0.8)
            embedding = my_umap.fit_transform(filtered_known_activations_per_stride[idx_stride], y=filtered_known_labels)

            # Transform the unknown classes data
            embedding_unknown = my_umap.transform(filtered_unknown_activations_per_stride[idx_stride])

            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(1, 1, 1)#, projection='3d')
            
            cmap = plt.cm.tab20(np.arange(40).astype(int))
            # Represent the known classes
            color_idx = 0  # For both known and unknown classes, to distinguish them
            for idx_cls, cl in enumerate(known_classes):
                plotEmbeddings = np.array([embedding[i,:] for i in range(len(embedding)) if filtered_known_labels[i]==cl])
                if len(plotEmbeddings)!=0:
                    ax.scatter(*plotEmbeddings.T, color=cmap[color_idx],label=CLASSES[cl],alpha=0.7)
                    color_idx = color_idx + 1

            # Represent the unknown classes
            for idx_cls, cl in enumerate(unknown_classes):
                plotEmbeddings_unk = np.array([embedding_unknown[i,:] for i in range(len(embedding_unknown)) if filtered_unknown_labels[i]==cl])
                if len(plotEmbeddings_unk) > 200:
                    ax.scatter(*plotEmbeddings_unk.T, color=cmap[color_idx], label=CLASSES[cl], alpha=0.7, marker='s')
                    color_idx = color_idx + 1
            
            plt.legend()
            fig.savefig(FIGS_PATH / f"umap_{model_folder_path.name}_stride_{(idx_stride+1)*8}.png")
            plt.close()


    else:  # All strides in one UMAP, with different markers each stride
        # Therefore, we have to make a big array with all the activations of all strides
        # We also have to have a big array with the labels of all strides, by repeating the array of labels 3 times
        my_umap = UMAP(n_components=2,n_neighbors=20, metric='cosine',min_dist=0.01,target_weight=0.8)
        embedding = my_umap.fit_transform(filtered_known_activations_per_stride[0], y=filtered_known_labels)
        raise NotImplementedError("Not implemented yet")
    
    # fig = plt.figure(figsize=(14, 10))
    # ax = fig.add_subplot(1, 1, 1)#, projection='3d')
    
    # cmap = plt.cm.tab20(np.arange(len(filterClassesNum)).astype(int))
    # indexC = 0
    # for c in filterClassesNum:
    #     plotEmbeddings = np.array([embedding[i,:] for i in range(len(embedding)) if filteredLabels[i]==c])
    #     if len(plotEmbeddings)!=0:
    #         ax.scatter(*plotEmbeddings.T, c=cmap[indexC],label=CLASSES[c],alpha=0.7)
    #     indexC = indexC + 1


if __name__ == "__main__":
    args = SimpleArgumentParser().parse_args()
    print(args)
    main()


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