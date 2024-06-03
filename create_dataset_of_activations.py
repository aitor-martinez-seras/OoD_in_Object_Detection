from pathlib import Path
from datetime import datetime
import argparse
import json
from typing import Literal, List

from tap import Tap

from ultralytics import YOLO
from ultralytics.yolo.utils.callbacks import tensorboard
from ultralytics.yolo.utils import yaml_load
from ultralytics.yolo.utils import DEFAULT_CFG_PATH

from data_utils import read_json, write_json, create_YOLO_dataset_and_dataloader, build_dataloader, create_TAO_dataset_and_dataloader
from ood_utils import ActivationsExtractor, configure_extra_output_of_the_model

# Constants
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.15


class SimpleArgumentParser(Tap):
    # Required arguments
    model_folder: str  # Which variant of the model YOLO to use
    device: int  # Device to use for training on GPU. Indicate more than one to use multiple GPUs. Use -1 for CPU.
    # Optional arguments
    dataset: str = "tao_coco"
    batch_size: int = 16  # Batch size.
    workers: int = 8  # Number of background threads used to load data.

    def configure(self):
        self.add_argument("-d", "--device")


def main():

    # Constants
    ROOT = Path().cwd()  # Assumes this script is in the root of the project
    NOW = datetime.now().strftime("%Y%m%d_%H%M")

    # Create the data extractor
    print('Creating the data extractor...')
    activations_extractor = ActivationsExtractor(agg_method='mean', iou_threshold_for_matching=IOU_THRESHOLD, min_conf_threshold_test=CONF_THRESHOLD)
    print('Data extractor created!')
    
    # Load model
    print('Loading model...')
    model = YOLO(model=ROOT / args.model_folder / 'weights' / 'best.pt')
    device = f'cuda:{args.device}'
    model.to(device)
    # Modify internal attributes of the model to obtain the desired outputs in the extra_item
    configure_extra_output_of_the_model(model, activations_extractor)
    print('Model loaded!')
    
    # Dataset selection
    yaml_file = f"{args.dataset}.yaml"
    train_dataset, train_dataloader = create_TAO_dataset_and_dataloader(
            yaml_file,
            args,
            data_split='train',
    )
    
    # Process data
    print('Extracting activations...')
    activations_extractor.iterate_data_to_extract_ind_activations_and_create_its_annotations(train_dataloader, model, device, split='train')
    print('Activations extracted!')

    val_dataset, val_dataloader = create_TAO_dataset_and_dataloader(
            yaml_file,
            args,
            data_split='train',
    )

    # Training
    

if __name__ == "__main__":
    args = SimpleArgumentParser().parse_args()
    print(args)
    main()