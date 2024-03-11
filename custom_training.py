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
    
    devices: List[int]  # Device to use for training on GPU. Indicate more than one to use multiple GPUs. Use -1 for CPU.
    epochs: int  # Number of epochs to train for.
    model: Literal["n", "s", "m", "l", "x"]  # Which variant of the model YOLO to use
    
    config: str = "config_for_tao_adam"
    dataset: str = "tao_coco"
    lr: float = 0.001  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    lrf: float = 0.01  # final learning rate (lr0 * lrf)
    cos_lr: bool = False  # Use cosine learning rate
    freeze_backbone: bool = False  # Freeze the backbone of the model
    batch_size: int = 16  # Batch size.
    val_every: int = 1  # Validate every n epochs
    workers: int = 10  # Number of background threads used to load data.
    close_mosaic: int = 20  # Close mosaic augmentation
    from_scratch: bool = False # Train the model from scratch, not pretrained in COCO.
    do_not_val_during_training: bool = False  # If passed, the model is NOT validated during training.
    model_path: str = ''  # Relative path to the model you want to use as a starting point. Deactivates using sizes.
    val_only: bool = False  # If passed, the model is only validated and not trained.
    owod_task: Literal["", "t1", "t2", "t3"] = ""  # OWOD task to train on. 

    def configure(self):
        self.add_argument("-e", "--epochs", required=False)
        self.add_argument("-m", "--model", required=False)
        self.add_argument("-cl_ms", "--close_mosaic")

    def process_args(self):
        if not self.epochs:
            assert self.val_only, "You must pass the number of epochs if you are not only validating."

        if self.model_path:
            print('Loading model from', self.model_path)
            print('Ignoring args --model --from_scratch')
            self.from_scratch = False
            self.model = ''
        else:
            if self.model == '':
                raise ValueError("You must pass a model size.")
        
        if 'owod' in self.dataset:
            if self.owod_task == '':
                raise ValueError("You must pass a OWOD task")
        

def main():

    # Workaround for the SSL error when downloading weights or datasets
    import ssl
    ssl._create_default_https_context = ssl._create_stdlib_context

    # Constants
    ROOT = Path().cwd()  # Assumes this script is in the root of the project
    NOW = datetime.now().strftime("%Y%m%d_%H%M")

    # Dataset selection
    yaml_file = f"{args.dataset}.yaml"
    if 'tao_coco' in args.dataset:
        project_name = 'TAO'
    elif 'owod' in args.dataset:
        project_name = 'OWOD'
    elif 'coco' in args.dataset:
        project_name = 'COCO'
    else:
        raise ValueError("The dataset must be one of the following: tao_coco, owod, coco")
    # dataset_info = yaml_load(ROOT / 'ultralytics/yolo/cfg' / yaml_file)
    # number_of_classes = len(dataset_info['names'])
    
    # Model selection
    if args.model_path:
        model_weights_path = ROOT / args.model_path
        model = YOLO(model_weights_path, task='detect')
        string_for_folder = "finetuned"
    else:
        if args.from_scratch:
            # build a new model from scratch
            model = YOLO(f"yolov8{args.model}.yaml", task='detect')
            string_for_folder = "from_scratch"
        else:
            # Use pretrained model on COCO
            model = YOLO(f"yolov8{args.model}.pt")  # https://docs.ultralytics.com/es/yolov5/tutorials/train_custom_data/#3-select-a-model
            string_for_folder = "pretrained"

    # Name of the folder to save the model, logs, etc.
    if project_name == 'OWOD':
        folder_name = f'{NOW}_{args.dataset}_{args.owod_task}_yolov8{args.model}_{string_for_folder}'
    else:
        folder_name = f'{NOW}_{args.dataset}_yolov8{args.model}_{string_for_folder}'

    # TODO
    if args.freeze_backbone:
        def freeze_layer(trainer):
            model = trainer.model
            num_freeze = 10
            print(f"Freezing {num_freeze} layers")
            freeze = [f'model.{x}.' for x in range(num_freeze)]  # layers to freeze 
            for k, v in model.named_parameters(): 
                v.requires_grad = True  # train all layers 
                if any(x in k for x in freeze): 
                    print(f'freezing {k}') 
                    v.requires_grad = False 
            print(f"{num_freeze} layers are freezed.")
        model.add_callback("on_train_start", freeze_layer)

    # TODO: Descrubrir como definir que los resultados de validacion se guarden en la propia carpeta del modelo
    #   y con fecha y nombre de contra que se ha probado
    if args.val_only:
        print("*** Starting validation ***")
        model.val(
            data=yaml_file,
            cfg=f"{args.config}.yaml",  # https://docs.ultralytics.com/es/usage/cfg/
            batch=args.batch_size,
            workers=args.workers,
            device=args.devices
        )
        # Save the arguments used to a file
        # TODO: Guardarlo en la carpeta del modelo original
        #args.save(ROOT / args.model_path / 'script_args.json')
    else:
        print("*** Starting training ***")
        model.train(
            data=yaml_file,
            cfg=f"{args.config}.yaml",  # https://docs.ultralytics.com/es/usage/cfg/
            project=f'runs_{project_name}',
            name=folder_name,
            lr0=args.lr,
            lrf=args.lrf,
            cos_lr=args.cos_lr,
            device=args.devices,
            epochs=args.epochs,
            batch=args.batch_size,
            mixup=0.0,
            close_mosaic=args.close_mosaic,
            workers=args.workers,
            plots=True,
            val=not args.do_not_val_during_training,
            val_every=args.val_every,
            owod_task=args.owod_task,
        )

        # Save the arguments used to a file
        args.save(ROOT / f'runs_{project_name}' / folder_name / 'script_args.json')


if __name__ == "__main__":
    #args = arg_parser().parse_args()
    args = SimpleArgumentParser().parse_args()
    #args_dict = vars(args)
    print(args)
    main()