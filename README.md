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
└── source_code_folder
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

# Download datasets

## TAO

Follow instructions inside ```datasets_utils/tao/instructions.md```

### Dataset description

Each frame of the annotated dataset is corresponding with the frame number ```(n * 30) - 1```