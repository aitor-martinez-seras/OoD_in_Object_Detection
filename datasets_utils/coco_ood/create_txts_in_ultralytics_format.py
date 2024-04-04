import os
from pathlib import Path
import json


def main():
    # To conver to the ultralytics format, we need to define a .txt where the image paths 
    # of the selected images are written from the root of the dataset.

    # Besides, the annotations have to be augmented with new annotations of unknown objects, therefore 
    # the json files must be loaded on runtime to augment the annotations.
    
    # Obtain the absolute path of the current file
    coco_ood_root_path = Path(os.path.abspath(__file__)).parent
    datasets_root_path = coco_ood_root_path.parent.parent.parent / "datasets"
    coco_dataset_root_path = datasets_root_path / "coco"
    voc_images_path = datasets_root_path / 'VOC/images/'

    voc_train2007_images_path = voc_images_path / 'train2007'
    voc_train2012_images_path = voc_images_path / 'train2012'
    voc_val2007_images_path = voc_images_path / 'val2007'
    voc_val2012_images_path = voc_images_path / 'val2012'
    voc_test2007_images_path = voc_images_path / 'test2007'
    voc_test2012_images_path = voc_images_path / 'test2012'


    # VOC
    train_voc_path = coco_ood_root_path / 'voc0712_train_all.json'
    train_pretest_path = coco_ood_root_path / 'voc0712_train_completely_annotation200.json'
    val_voc_path = coco_ood_root_path / 'val_coco_format.json'
    with open(train_voc_path, 'r') as f:
        train_voc_ann = json.load(f)
    with open(train_pretest_path, 'r') as f:
        train_pretest_ann = json.load(f)
    with open(val_voc_path, 'r') as f:
        val_voc_ann = json.load(f)

    # COCO OOD
    print('Creating txt for OOD images...')
    coco_ood_txt_path = coco_dataset_root_path / 'val_ood.txt'
    coco_ood_ann_path =  coco_ood_root_path / 'instances_val2017_coco_ood.json'
    with open(coco_ood_ann_path, 'r') as f:
        coco_ood_ann = json.load(f)
    with open(coco_ood_txt_path, 'w') as f:
        for image in coco_ood_ann['images']:
            f.write(f"./images/val2017/{image['file_name']}\n")
    print('Done!')

    # COCO Mixed
    # As both IND and OOD .json files contain same images,
    # we can use only one of the two .json files for the txt creation
    print('Creating txt for mixed images...')
    coco_mixed_txt_path = coco_dataset_root_path / 'val_mixed.txt'
    coco_mixed_ind_ann_path = coco_ood_root_path / 'instances_val2017_mixed_ID.json'
    coco_mixed_ood_ann_path = coco_ood_root_path / 'instances_val2017_mixed_OOD.json'
    with open(coco_mixed_ind_ann_path, 'r') as f:
        coco_mixed_ind_ann = json.load(f)
    with open(coco_mixed_ood_ann_path, 'r') as f:
        coco_mixed_ood_ann = json.load(f)
    with open(coco_mixed_txt_path, 'w') as f:
        for image in coco_mixed_ind_ann['images']:
            f.write(f"./images/val2017/{image['file_name']}\n")
    print('Done!')


if __name__ == '__main__':
    main()
