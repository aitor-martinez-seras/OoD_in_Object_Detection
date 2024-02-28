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
    dataset: str = "tao_coco"
    batch_size: int = 16  # Batch size.
    workers: int = 8  # Number of background threads used to load data.
    close_mosaic: int = 20  # Close mosaic augmentation
    model_path: str = ''  # Path to the model you want to use as a starting point. Deactivates using sizes.
    from_scratch: bool = False # Train the model from scratch, not pretrained in COCO.

    def configure(self):
        self.add_argument("-e", "--epochs")
        self.add_argument("-m", "--model")
        self.add_argument("-d", "--devices")
        self.add_argument("-cl_ms", "--close_mosaic")


# def arg_parser():
#     parser = argparse.ArgumentParser()
#     # Required arguments
#     parser.add_argument("-e", "--epochs", type=int, required=True, help="Number of epochs to train for.")
#     parser.add_argument("--model", required=True, choices=["n", "s", "m", "l", "x"], help="Which variant to use")
#     parser.add_argument("-d", "--devices", type=int, nargs="+", default="0",
#                         help="Device to use for training on GPU. Indicate more than one to use multiple GPUs. Use -1 for CPU.")
    
#     # Optional arguments
#     parser.add_argument("--dataset", type=str, default="tao_coco")
#     parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
#     parser.add_argument("--workers", type=int, default=8, help="Number of background threads used to load data.")
#     parser.add_argument("-cl_ms", "--close_mosaic", type=int, default=20, help="Close mosaic augmentation")
#     parser.add_argument("--model_path", type=str, help="Path to the model you want to use as a starting point. Deactivates using sizes.")
#     parser.add_argument("--from_scratch", action="store_true", help="Train the model from scratch, not pretrained in COCO.")
#     return parser


def main():

    # Workaround for the SSL error
    import ssl
    ssl._create_default_https_context = ssl._create_stdlib_context

    # Constants
    ROOT = Path().cwd()  # Assumes this script is in the root of the project
    NOW = datetime.now().strftime("%Y%m%d_%H%M")
    
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
    
    # Dataset selection
    yaml_file = f"{args.dataset}.yaml"

    # Training
    model.train(
        data=yaml_file,
        cfg="config_for_tao_adam.yaml",  # https://docs.ultralytics.com/es/usage/cfg/
        device=args.devices,
        epochs=args.epochs,
        batch=args.batch_size,
        mixup=0.0,
        close_mosaic=args.close_mosaic,
        workers=args.workers,
        name=folder_name,
    )
    args.save(ROOT / folder_name / 'script_args.json')
    # with open(ROOT / folder_name / 'script_args.json', 'w') as json_file:
    #     json.dump(args_dict, json_file, indent=4)

# path = model.export(format="onnx")  # export the model to ONNX format


if __name__ == "__main__":
    #args = arg_parser().parse_args()
    args = SimpleArgumentParser().parse_args()
    #args_dict = vars(args)
    print(args)
    main()