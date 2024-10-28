# Disclaimer

JSON files downloaded from [UnSniffer](https://github.com/Went-Liang/UnSniffer/tree/main)

# Usage 

Execute [create_txts_in_ultralytics_format.py](create_txts_in_ultralytics_format.py) to create the `.txt` files with the image paths in the Ultralytics format.

# Explanation of the files

## Pascal VOC

- `voc0712_train_all.json`: used to train the detector (backbone of UnSniffer pretrained in ImageNet)
- `val_coco_format.json`: used to evaluate the detector
- `voc0712_train_completely_annotation200.json`: used to define the thresholds

## COCO

- `instances_val2017_coco_ood.json`: split with __ONLY__ OOD or unknown instances

This file contains both unknown already annotated (from class 21 to 80 from coco) and unknown instances that are annotated by the UnSniffer community. All of them are annotated as class 81. The annotations of the already annotated instances are first in the list.

- `instances_val2017_mixed_OOD.json`: annotations of OOD instances in the split with mixed OOD and ID instances
- `instances_val2017_mixed_ID.json`: annotations of ID instances in the split with mixed OOD and ID instances

__WARNING__: The annotations in mixed OOD json have the problem that contain some image ids in the "annotations" key that are not in the "images" key. I assume this image IDs are skipped when loading the dataset.


# Implementation details

Annotations of bbox are in [x, y, width, height] format, therefore should be converted to [cx, cy, h, w] format and normalized.

1. Create the `.txt` files with the image paths [create_txts_in_ultralytics_format.py](create_txts_in_ultralytics_format.py)
2. Create the YAML files with the OWOD names and the mapping from COCO OOD classes order (not the same as standard COCO) to OWOD names. This files are already in the repository (ultralytics/yolo/cfg).
3. In the [dataset.py](ultralytics/yolo/data/dataset.py), make the function load the json files from this folder and replace the labels loaded from the validation split of COCO with the annotations from the json files. It is important to note that classes must be mapped from the COCO OOD classes to the OWOD classes (case of Mixed dataset) and that bounding boxes must be converted to the [cx, cy, h, w] format and normalized.
