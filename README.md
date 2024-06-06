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

### Dataset description

Each frame of the annotated dataset is corresponding with the frame number ```(n * 30) - 1```

# YOLO Sizes an architecture

## Sizes
![Yolo_sizes_image](ultralytics/assets/yolov8_sizes.png)

## Architecture
![Yolo_arch](ultralytics/assets/yolov8_arch.jpg)