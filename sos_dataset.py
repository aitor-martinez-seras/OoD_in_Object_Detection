import json
from typing import Union, List, Optional, Callable, Iterable, TypeVar
from pathlib import Path

import numpy as np
import torch
from natsort import natsorted
from torchvision.ops import box_convert
from torchvision.transforms import Compose
from PIL import Image

from ultralytics.yolo.data import BaseDataset
from ultralytics.yolo.data.augment import Format, LetterBox, Instances
from ultralytics.yolo.utils import DEFAULT_CFG
from data_utils import segmentation_to_bbox 


class SOS_BaseDataset(BaseDataset):

    def __init__(self,
                 imgs_path,
                 ann_path=None,
                 imgsz=640,
                 cache=False,
                 augment=False,
                 hyp=DEFAULT_CFG,  # TODO: Aqui tengo que poner los hiperparametros desde la DEFAULT_CFG
                 prefix='',
                 rect=False,
                 batch_size=16,
                 stride=32,
                 pad=0.5,
                 single_cls=False,
                 classes=None,
                 fraction=1.0):
                 
        self.ann_path = ann_path
        super().__init__(imgs_path, imgsz, cache, augment, hyp, prefix, rect, batch_size, stride, pad, single_cls, classes, fraction)

    def update_labels_info(self, label):
        """custom your label format here."""
        # NOTE: cls is not with bboxes now, classification and semantic segmentation need an independent cls label
        # we can make it also support classification and semantic segmentation by add or remove some dict keys there.
        bboxes = label.pop('bboxes')
        segments = label.pop('segments')
        keypoints = label.pop('keypoints', None)
        bbox_format = label.pop('bbox_format')
        normalized = label.pop('normalized')
        label['instances'] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    def build_transforms(self, hyp=None):
        """Users can custom augmentations here
        like:
            if self.augment:
                # Training transforms
                return Compose([])
            else:
                # Val transforms
                return Compose([])
        """
        if self.augment:
                # Training transforms
                return Compose([])
        else:
            # Val transforms
            return Compose([
                LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False),
                Format(bbox_format='xywh',
                        normalize=True,
                        return_mask=False,
                        return_keypoint=False,
                        batch_idx=True,
                        mask_ratio=hyp.mask_ratio,
                        mask_overlap=hyp.overlap_mask)
                ])


    def get_labels(self):
        """Users can custom their own format here.
        Make sure your output is a list with each element like below:
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes, # xywh
                segments=segments,  # xy
                keypoints=keypoints, # xy
                normalized=True, # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
        """
        with open(self.ann_path, 'r') as f:
            anns = json.load(f)
        imgs_info_per_id = {}
        for i, ann in enumerate(anns['images']):
            imgs_info_per_id[ann['id']] = {
                "file_name": ann['file_name'],
                "height": ann['height'],
                "width": ann['width'],
            }

        labels = []
        for i, ann in enumerate(anns['annotations']):
            current_img_info = imgs_info_per_id[ann['image_id']]
            boxes = np.array(ann['bbox'], dtype=np.float64)
            if len(boxes.shape) == 1:
                boxes = boxes[np.newaxis, :]
            labels.append(
                {
                    'im_file': current_img_info['file_name'],
                    'shape': (current_img_info['height'], current_img_info['width']),
                    'cls': np.array(ann['category_id'], ndmin=1),
                    'bboxes': boxes,
                    'segments': ann['segmentation'],
                    'keypoints': None,
                    'normalized': False,
                    'bbox_format': 'xywh'
                }
            )

        return labels

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == 'img':
                value = torch.stack(value, 0)
            if k in ['masks', 'keypoints', 'bboxes', 'cls']:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch['batch_idx'] = list(new_batch['batch_idx'])
        for i in range(len(new_batch['batch_idx'])):
            new_batch['batch_idx'][i] += i  # add target image index for build_targets()
        new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)
        return new_batch



def convert_target(target_list, targets_basenames):
    target = []
    j=0
    for i in range(len(targets_basenames)):
        if targets_basenames[i] == True:
            target.append(target_list[j])
            j +=1
        else:
            target.append(None)
    return target

class SOS():
    """
    Dataset class for loading the Street Obstacle Sequences dataset with target types of the original dataset.
    It is being used just to load labels and create the annotations json file.
    Args:
        root (string): Root directory of dataset
        sequences (string, optional): The image sequences to load
        target_type (string or list, optional): Type of target to use, choose from ("semantic_ood", "instance_ood", "depth_ood", "semantic").
        transforms (callable, optional): A function/transform that takes input sample and its target as entry and returns a transformed version.
    """
    def __init__(self, root: str, sequences: str = ["all"],  target_type: Union[List[str], str] = "semantic_ood", transforms: Optional[Callable] = None):
        self.root = root
        self.images = []
        self.all_images = []
        self.targets_semantic_ood = []
        self.targets_instance_ood = []
        self.targets_depth_ood = []
        self.targets_semantic = []
        self.basenames = []
    
        self.all_basenames = []
        self.ood_id = 254
        self.target_type = target_type
        self.transforms = transforms
        self.ood_classes = np.arange(244, 255)
        self.id_dict = {'sequence_001': [1], 
                        'sequence_002': [1],
                        'sequence_003': [1],
                        'sequence_004': [1],
                        'sequence_005': [1],
                        'sequence_006': [1],
                        'sequence_007': [1],
                        'sequence_008': [1],
                        'sequence_009': [1],
                        'sequence_010': [1],
                        'sequence_011': [1],
                        'sequence_012': [1],
                        'sequence_013': [1],
                        'sequence_014': [1],
                        'sequence_015': [1, 2],
                        'sequence_016': [1, 2],
                        'sequence_017': [1, 2],
                        'sequence_018': [1, 2],
                        'sequence_019': [1, 2],
                        'sequence_020': [1, 2]}

        
        if not isinstance(target_type, list):
            self.target_type = [target_type]
        available_target_types = ("semantic_ood", "instance_ood", "depth_ood", "semantic")
        if not target_type in available_target_types:
            raise NameError(f"Unknown target_type '{target_type}'. Valid values are {{{available_target_types}}}.")
        # [verify_str_arg(value, "target_type", available_target_types) for value in self.target_type]
        
        if sequences is None or "all" in [str(s).lower() for s in sequences]:
            self.sequences = []
            for sequence in (Path(self.root) / "raw_data").glob("sequence*"):
                self.sequences.append(str(sequence.name))
        elif all(isinstance(s, int) for s in sequences):
            self.sequences = []
            for s in sequences:
                self.sequences.append("sequence_" + str(s).zfill(3))
        else:
            self.sequences = sequences
        self.sequences = natsorted(self.sequences)
        
        for sequence in self.sequences:
            sequence_images_dir = Path(self.root) / "raw_data" / sequence
            sequence_semantic_ood_dir = Path(self.root) / "semantic_ood" / sequence
            sequence_instance_ood_dir = Path(self.root) / "instance_ood" / sequence
            sequence_depth_ood_dir = Path(self.root) / "depth_ood" / sequence
            sequence_semantic_dir = Path(self.root) / "semantic" / sequence
            
            sequence_basenames = []
            for file_path in sequence_semantic_ood_dir.glob("*_semantic_ood.png"):
                sequence_basenames.append(str(Path(sequence) / f"{file_path.stem}").replace("_semantic_ood", ""))
            sequence_basenames = natsorted(sequence_basenames)
            for basename in sequence_basenames:
                self.basenames.append(basename)
                self.images.append(str(sequence_images_dir / f"{Path(basename).stem}_raw_data.jpg"))
                self.targets_semantic_ood.append(str(sequence_semantic_ood_dir / f"{Path(basename).stem}_semantic_ood.png"))
                self.targets_instance_ood.append(str(sequence_instance_ood_dir / f"{Path(basename).stem}_instance_ood.png"))
                self.targets_depth_ood.append(str(sequence_depth_ood_dir / f"{Path(basename).stem}_depth_ood.png"))
                self.targets_semantic.append(str(sequence_semantic_dir / f"{Path(basename).stem}_semantic.png"))
                

            for file_path in sequence_images_dir.glob("*.jpg"):
                self.all_images.append(str(file_path))
                self.all_basenames.append(str(Path(sequence) / file_path.stem.replace('_raw_data','')))
                
        self.all_images = natsorted(self.all_images)
        self.all_basenames = natsorted(self.all_basenames)
        self.targets_basenames = [ self.all_basenames[i] in self.basenames for i in range(len(self.all_basenames))]
        self.targets_semantic_ood = convert_target(self.targets_semantic_ood, self.targets_basenames)
        self.targets_instance_ood = convert_target(self.targets_instance_ood, self.targets_basenames)
        self.targets_depth_ood = convert_target(self.targets_depth_ood, self.targets_basenames)
        self.targets_semantic = convert_target(self.targets_semantic, self.targets_basenames)
        #self.targets_bbox = convert_target(self.targets_bbox, self.targets_basenames)


def create_annotations_json_for_sos_dataset(sos_dataset: SOS, mode: str, coco: bool):
    json_file = {
        "info": {
            "year": 2022,
            "version": "1.0",
            "description": "Street Obstacle Sequences Dataset",
            "contributor": "Kira Maag; Ruhr University Bochum, Germany",
            "url": "",
            "date_created": "2022/10/04"
        },
        "licenses": [
            {
                "url": "",
                "id": 1,
                "name": "License"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [{
            "id": 1,
            "name": "OOD",
            "supercategory": "OOD",
        }]
    }

    # image structure:
    # image{
    #     "id": int,
    #     "width": int,
    #     "height": int,
    #     "file_name": str,
    #     "license": int,
    #     "flickr_url": str,
    #     "coco_url": str,
    #     "date_captured": datetime,
    # }

    if mode == 'val':

        bbox_idx = 0
        img_idx = 0
        total_number_of_annotated_imgs = len(sos_dataset.targets_instance_ood) - sos_dataset.targets_instance_ood.count(None)
        
        for idx, seg_img_path in enumerate(sos_dataset.targets_instance_ood):

            width, height = 0, 0
            
            # Fill annotation information
            if seg_img_path is not None:
                
                # Print progress
                if img_idx % 100 == 0:
                    print(f"Image {img_idx} of {total_number_of_annotated_imgs}")
                
                # Image path
                seg_img_path = Path(seg_img_path)
                if coco:
                    img_path = Path("raw_data", seg_img_path.parts[-2], seg_img_path.parts[-1].replace("_instance_ood.png", "_raw_data.jpg"))
                else:
                    img_path = Path(seg_img_path.parts[-2], seg_img_path.parts[-1].replace("_instance_ood.png", "_raw_data.jpg"))

                # Load annotation image
                segmentation_img = Image.open(seg_img_path)
                width, height = segmentation_img.size
                segmentation_img = np.array(segmentation_img)
                instance_ids = np.unique(segmentation_img)

                for inst_id in instance_ids:

                    if inst_id != 255:
                        bbox_xyxy = segmentation_to_bbox(segmentation_img, inst_id)
                        json_file["annotations"].append({
                            "id": bbox_idx,
                            "image_id": img_idx,
                            "category_id": 0,
                            "segmentation": [],
                            "area": 0,  # Segmentation area
                            "bbox": box_convert(torch.tensor(bbox_xyxy), "xyxy", "xywh").tolist(),
                            "iscrowd": 0
                        })
                        bbox_idx += 1

                # Fill image information
                json_file["images"].append({
                    "id": img_idx,
                    "width": width,
                    "height": height,
                    "file_name": img_path.as_posix(),
                    "license": 1,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": "2022/10/04"
                })
                img_idx += 1
        if coco:
            with open(f'val_coco_annotations.json', 'w') as outfile:
                            json.dump(json_file, outfile)
        else:
            with open(f'custom_datasets/val_annotations.json', 'w') as outfile:
                json.dump(json_file, outfile)


    elif mode == 'test':
        # Aqui tengo que tener codigo para generar las annotaciones de tal forma que se incluyan
        # todas las imagenes en el apartado de "images" pero que en el apartado de "annotations" 
        # solo se incluyan las que tienen anotaciones realmente, para asi poder iterar en un futuro
        # sobre todas las imagenes y no solo las anotadas
        raise NotImplementedError


if __name__ == '__main__':
    create_annotations_json_for_sos_dataset(
        sos_dataset=SOS(
            root='/home/tri110414/nfs_home/datasets/street_obstacle_sequences/',
            sequences=["all"],
            target_type="instance_ood",
            transforms=None
        ),
        mode='val',
        coco=False
    )