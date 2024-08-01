# Introduction

This is the code for the paper ______. 

The code is based on an early version of the Ultralytics YOLOv8 repository. The more updated version of the libray can be found [here](https://github.com/ultralytics/ultralytics).

# Instalation of the package

To install the package:

1. Install torch>=1.8 and torchvision>=0.8.1, with CUDA support preferably
2. ```pip install -r requirements.txt```
3. If you want to use CLI commands, use ```pip install -e .```


# Basic structure

```bash
├── datasets
│   ├── coco
│   └── TAO
└── < source_code_folder >
    ├── datasets_utils
    ├── data_utils.py
    ├── examples
    ├── ultralytics
    ├── ultralytics.egg-info
    ├── venv-yolo
    ├── visualization_utils.py
    ...
    └── yolov8n.pt
```

Datasets must be outside the source code folder

# Warnings

The code has been only tested on Linux and some of the code may not work, as some parts are not OS agnostic due to the usage of the `/` for splitting paths.

# Download datasets

## OWOD (Open World Object Detection)

Follow instructions in [datasets_utils/owod/instructions.md](datasets_utils/owod/instructions.md).

## COCO OOD and COCO Mixed

Follow instructions in [datasets_utils/coco_ood/instructions.md](datasets_utils/coco_ood/instructions.md).

# Usage

The model obtaining the results of the paper can be downloaded from [this OneDrive link](https://tecnalia365-my.sharepoint.com/:u:/g/personal/aitor_martinez_tecnalia_com/EbHfJyzcQy9Kr1-h2SOP5MsBLaQfLmMRbWtkcZxDHTIDsw?e=4D9VYi).

## Replicate results

1) Train a model using the ```replicate/train.sh```. The arguments are the same to the ones used for the paper.

2) Run the bash scripts inside the  ```replicate/benchmarks.sh``` to obtain the results.

3) To obtain the figures of the paper, run the ```process_results.ipynb```. The code expects the following structure:
    
    ```bash
    < source_code_folder >
    ├── results
        ├── fmap_method
        │   └── < .csv files from the vanilla FMap method >
        ├── fmap_method_SDR
        │   └── < .csv files from the SDR FMap method >
        ├── fmap_method_EUL
        │   └── < .csv files from the EUL FMap method >
        ├── logits_methods
        │   └── < .csv files from the post-hoc or logits methods >
        └── fusion_methods
            └── < .csv files from the fusion methods >
    ```


## Using the OOD methods

Examples of usage are provided in the ```scripts/ood_evaluation.sh``` file.

## Training a YOLO model

Examples are provided ```scripts/train_owod.sh``` file.

# YOLO Sizes an architecture

## Sizes
![Yolo_sizes_image](ultralytics/assets/yolov8_sizes.png)

## Architecture
![Yolo_arch](ultralytics/assets/yolov8_arch.jpg)