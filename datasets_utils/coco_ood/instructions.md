# Disclaimer

JSON files downloaded from [UnSniffer](https://github.com/Went-Liang/UnSniffer/tree/main)

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



# Implementation

Annotations of bbox are in [x, y, width, height] format, therefore should be converted to [cx, cy, h, w] format and normalized.

1. Create the `.txt` files with the image paths
2. When used the YAML file of the 