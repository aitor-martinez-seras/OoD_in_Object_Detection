from pathlib import Path
from datetime import datetime
from typing import Literal, List

from tap import Tap

from ultralytics import YOLO
from ultralytics.utils.callbacks import tensorboard
from ultralytics.utils import yaml_load
from ultralytics.utils import DEFAULT_CFG_PATH


class SimpleArgumentParser(Tap):
    
    devices: List[int]  # Device to use for training on GPU. Indicate more than one to use multiple GPUs. Use -1 for CPU.
    model: str = "yolov8"  # Model to use. Options: yolov5, yolov6, yolov8, custom_yolov8. Default: yolov8
    model_size: Literal["n", "s", "m", "l", "x"]  # Which variant of the model YOLO to use
    dataset: str = "tao_coco"
    # Hyperparameters
    epochs: int  # Number of epochs to train for.
    config: str = "config_for_tao_adam"
    lr: float = 0.001  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    lrf: float = 0.01  # final learning rate (lr0 * lrf)
    cos_lr: bool = False  # Use cosine learning rate
    freeze_backbone: bool = False  # Freeze the backbone of the model
    batch_size: int = 16  # Batch size.
    close_mosaic: int = 20  # Close mosaic augmentation
    from_scratch: bool = False # Train the model from scratch, not pretrained in COCO.
    imagenet_pretrained_backbone: bool = False  # Use the pretrained backbone from ImageNet. Assumes rest of the model is from scratch.
    # Training options
    model_path: str = ''  # Relative path to the model you want to use as a starting point. Deactivates using model_size.
    workers: int = 10  # Number of background threads used to load data.
    val_every: int = 1  # Validate every n epochs
    do_not_val_during_training: bool = False  # If passed, the model is NOT validated during training.
    val_only: bool = False  # If passed, the model is only validated and not trained.
    # OWOD
    owod_task: Literal["", "t1", "t2", "t3"] = ""  # OWOD task to train on.
    

    def configure(self):
        self.add_argument("-e", "--epochs", required=False)
        self.add_argument("-m", "--model", required=False)
        self.add_argument("-cl_ms", "--close_mosaic")
        self.add_argument("-im_pt_bck", "--imagenet_pretrained_backbone", action="store_true")

    def process_args(self):
        if not self.epochs:
            assert self.val_only, "You must pass the number of epochs if you are not only validating."
        
        if self.model:
            # Assert that the model size will work with the model passed
            if self.model_size:
                # import re
                # try:
                #     yaml_path = f"{self.model}{self.model_size}.yaml"
                #     re.search(r'yolov\d+([nslmx])', Path(yaml_path).stem).group(1)
                # except AttributeError:
                #     raise ValueError("The YAML file name passed will not work with the model size passed. It must have the substring 'yolov8{size}' in the name.")
                pass

        if self.model_path:
            print('Loading model from', self.model_path)
            print('Ignoring args --model --from_scratch')
            self.from_scratch = False
            self.model_size = ''
        else:
            if self.model_size == '':
                raise ValueError("You must pass a model size.")
        
        if 'owod' in self.dataset:
            if self.owod_task == '':
                raise ValueError("You must pass a OWOD task")

        if self.imagenet_pretrained_backbone:
            assert self.model == "yolov8", "The pretrained backbone from ImageNet is only available for YOLOv8"
            self.from_scratch = True
            print("Using the pretrained backbone from ImageNet. The rest of the model is trained from scratch.")
        

def select_number_of_classes(owod_task: str) -> int:
    if owod_task == 't1':
        return 20
    elif owod_task == 't2':
        return 40
    elif owod_task == 't3':
        return 60
    else:
        return 0  # 0 means that the number of classes is not going to be changed

def main():

    # Workaround for the SSL error when downloading weights or datasets
    import ssl
    ssl._create_default_https_context = ssl._create_stdlib_context

    # Constants
    ROOT = Path(__file__).parent  # Assumes this script is in the root of the project
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
            model = YOLO(f"{args.model}{args.model_size}.yaml", task='detect')
            string_for_folder = "from_scratch"
        else:
            # Use pretrained model on COCO
            model = YOLO(f"{args.model}{args.model_size}.pt")  # https://docs.ultralytics.com/es/yolov5/tutorials/train_custom_data/#3-select-a-model
            string_for_folder = "pretrained"

    # TODO: Add the option to plug in the Imagenet pretrained backbone in YOLOv8
    if args.imagenet_pretrained_backbone:
        imagenet_pretrained_yolov8 = YOLO(f"yolov8{args.model_size}-cls.pt")
        imagenet_model_weights = imagenet_pretrained_yolov8.model.model[:7].state_dict()  # We can only load until layer 6
        load_process_info = model.model.model.load_state_dict(imagenet_model_weights, strict=False)
        print("Pretrained backbone from ImageNet plugged in YOLOv8")

    # Name of the folder to save the model, logs, etc.
    if project_name == 'OWOD':
        folder_name = f'{NOW}_{args.dataset}_{args.owod_task}_{args.model}{args.model_size}_{string_for_folder}'
    else:
        folder_name = f'{NOW}_{args.dataset}_{args.model}{args.model_size}_{string_for_folder}'

    # Select the number of classes for OWOD
    number_of_classes = select_number_of_classes(args.owod_task)

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
            number_of_classes=number_of_classes,
        )

        # Save the arguments used to a file
        args.save(ROOT / f'runs_{project_name}' / folder_name / 'script_args.json')


if __name__ == "__main__":
    #args = arg_parser().parse_args()
    args = SimpleArgumentParser().parse_args()
    #args_dict = vars(args)
    print(args)
    main()