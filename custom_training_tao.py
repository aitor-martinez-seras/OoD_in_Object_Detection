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


class SimpleArgumentParser(Tap):
    # Required arguments
    epochs: int  # Number of epochs to train for.
    model: Literal["n", "s", "m", "l", "x"]  # Which variant of the model YOLO to use
    devices: List[int]  # Device to use for training on GPU. Indicate more than one to use multiple GPUs. Use -1 for CPU.
    # Optional arguments
    config: str = "config_for_tao_adam"
    dataset: str = "tao_coco"
    lr: float = 0.001  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    lrf: float = 0.01  # final learning rate (lr0 * lrf)
    cos_lr: bool = False  # Use cosine learning rate
    batch_size: int = 16  # Batch size.
    val_every: int = 1  # Validate every n epochs
    workers: int = 10  # Number of background threads used to load data.
    close_mosaic: int = 20  # Close mosaic augmentation
    from_scratch: bool = False # Train the model from scratch, not pretrained in COCO.
    do_not_val_during_training: bool = False  # If passed, the model is NOT validated during training.
    model_path: str = ''  # Path to the model you want to use as a starting point. Deactivates using sizes.

    def configure(self):
        self.add_argument("-e", "--epochs")
        self.add_argument("-m", "--model")
        self.add_argument("-cl_ms", "--close_mosaic")


def main():

    # Workaround for the SSL error
    import ssl
    ssl._create_default_https_context = ssl._create_stdlib_context

    # Constants
    ROOT = Path().cwd()  # Assumes this script is in the root of the project
    NOW = datetime.now().strftime("%Y%m%d_%H%M")

    # Dataset selection
    yaml_file = f"{args.dataset}.yaml"
    project_name = 'TAO' if 'tao' in args.dataset else 'COCO'
    # dataset_info = yaml_load(ROOT / 'ultralytics/yolo/cfg' / yaml_file)
    # number_of_classes = len(dataset_info['names'])
    
    # TODO: Future work
    if args.model_path:
        raise NotImplementedError("Loading a model from a path is not implemented yet.")
        model_to_load = args.model_path

    # Model selection
    if args.from_scratch:
        # build a new model from scratch
        model = YOLO(f"yolov8{args.model}.yaml", task='detect')
        string_for_folder = "from_scratch"
    else:
        # Use pretrained model on COCO
        model = YOLO(f"yolov8{args.model}.pt")  # https://docs.ultralytics.com/es/yolov5/tutorials/train_custom_data/#3-select-a-model
        string_for_folder = "pretrained"

    # Name of the folder to save the model, logs, etc.
    folder_name = f'{NOW}_{args.dataset}_yolov8{args.model}_{string_for_folder}'

    # Training
    model.train(
        data=yaml_file,
        cfg=f"{args.config}.yaml",  # https://docs.ultralytics.com/es/usage/cfg/
        project=f'runs_{project_name}',
        lr0=args.lr,
        lrf=args.lrf,
        cos_lr=args.cos_lr,
        device=args.devices,
        epochs=args.epochs,
        batch=args.batch_size,
        mixup=0.0,
        close_mosaic=args.close_mosaic,
        workers=args.workers,
        name=folder_name,
        plots=True,
        val=not args.do_not_val_during_training,
        val_every=args.val_every,
    )

    # Save the arguments used to a file
    args.save(ROOT / 'runs_TAO' / folder_name / 'script_args.json')


if __name__ == "__main__":
    #args = arg_parser().parse_args()
    args = SimpleArgumentParser().parse_args()
    #args_dict = vars(args)
    print(args)
    main()