# OWOD (Open World Object Detection)

## How to create the OWOD dataset

1. Download COCO and VOC datasets as in the download script in their YAML files ([ultralytics/yolo/cfg/coco.yaml](ultralytics/yolo/cfg/coco.yaml) and [ultralytics/yolo/cfg/VOC.yaml](ultralytics/yolo/cfg/VOC.yaml)), in the download key.

2. Execute [datasets_utils/owod/owod_dataset_utils.py](datasets_utils/owod/owod_dataset_utils.py). This creates the OWOD folder in the datasets folder, with the .txt that contain the paths for the images and the labels of COCO and VOC datasets that form the OWOD dataset.

## Explanation

The dataset was defined by Towards Open World Open Detection ([https://github.com/JosephKJ/OWOD](https://github.com/JosephKJ/OWOD)), and used in further papers related with this paradigm. The dataset is created by taking the COCO and VOC datasets with different tasks (T1,T2,T3,T4), each adding 20 new classes. The training set of T1 is composed by all the VOC dataset (2007 and 2012), and then the COCO dataset is used as test, as it contains new classes that can be considered unknown annotated.