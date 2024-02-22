### CODE FOR TAO DATASET CLASS ###
import json
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict, defaultdict
import time
import itertools

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import Compose
from PIL import Image

from ultralytics.yolo.data import BaseDataset
#from ultralytics.yolo.data.augment import Format, LetterBox, Instances
from .augment import Compose, Format, Instances, LetterBox, v8_transforms
from ultralytics.yolo.utils import DEFAULT_CFG
from .utils import HELP_URL, IMG_FORMATS

# Mapping from COCO to TAO classes (keys are COCO ids and values are TAO ids). 
# Obtained from https://github.com/YangLiu14/Open-World-Tracking/blob/main/datasets/coco_id2tao_id.json 
COCO_TO_TAO_MAPPING = {"13": 91, "34": 58, "33": 621, "49": 747, "8": 118, "51": 221, "1": 95, "73": 126, "79": 1122, "27": 729, "48": 926, "61": 1117, "11": 1038, "40": 1215, "74": 276, "21": 78, "75": 1162, "68": 699, "55": 185, "47": 13, "59": 79, "30": 982, "60": 371, "65": 896, "14": 99, "63": 642, "6": 1135, "64": 717, "53": 829, "70": 1115, "67": 235, "0": 805, "32": 41, "10": 452, "25": 1155, "7": 1144, "43": 625, "35": 60, "23": 502, "4": 4, "12": 779, "57": 1001, "38": 1099, "24": 34, "46": 45, "45": 139, "36": 980, "39": 133, "16": 382, "29": 480, "50": 154, "20": 429, "2": 211, "54": 392, "28": 36, "41": 347, "78": 544, "37": 1057, "9": 1132, "62": 1097, "44": 1018, "17": 579, "3": 714, "22": 1229, "15": 229, "77": 1091, "26": 35, "71": 979, "66": 299, "5": 174, "42": 475, "56": 237, "72": 428, "76": 937, "18": 961, "58": 852, "31": 993, "19": 81}

# Explanation of customization:
#   Apart from that, we need to modify the following files and functions:
#       - ultralytics/yolo/data/__init__.py: We need to add the import of this file to the __init__.py file
#       - ultralytics/yolo/data/build.py: We need to create build_tao_dataset function and add the import of TAODataset to the file
#       - ultralytics/yolo/v8/detect/train.py: We need to add the import of build_tao_dataset and modify the build_dataset function to use it

#   To create the TAODataset class, we inherit from BaseDataset, which contains the basic structure of a dataset class in Ultralytics YOLO. 
#   It contains the logic to call the functions that get image paths and labels, to load images from file paths... we need to modify or create the
#   following functions to customize the dataset class to our needs:
#       - We need to customize the get_labels function to load the labels from the .json file
#       - We need to customize the update_labels_info function to change the format of the labels (In our case we are using the same as YOLODataset)
#       - We need to customize the get_img_files function to load the image paths from the .json file
#       - We need to customize the build_transforms function to add the transforms we want to use (In our case we are using the same as YOLODataset)
#       - We need to customize the collate_fn function to change the way the batches are built (In our case we are using the same as YOLODataset)
#       - createIndex, _isArrayLike and getAnnIds are functions from the cocoapi that are used to load the annotations from the .json file
#
#   It is CRUCIAL that the self.im_files obtained from get_img_files function are in the same order and in the same number as the labels 
#   obtained from get_labels function, as the images and labels are loaded by get_image_and_label function from BaseDataset class, which takes 
#   an index parameter, and loads the image (using load_image) and label (in the function get_image_and_label itself) from the index-th element
#   of self.im_files and self.labels, respectively.
#   If we want to have image with no annotations included, empty annotations should be added.

#####################################################
# IMPORTANT NOTE: Bboxes MUST be in format CXCYWH, which in this version of Ultralytics YOLO is called "xywh" format
# Bboxes MUST be NORMALIZED in order to properly work
#####################################################

class TAODataset(BaseDataset):

    def __init__(self, *args, data=None, use_segments=False, use_keypoints=False, **kwargs):
        self.use_segments = use_segments
        self.use_keypoints = use_keypoints
        # # TODO: Añadir la opcion de ponerle las clases aqui a partir de el .json de las annotations
        # with open(Path(data["path"]) / data["train"], 'r') as f:
        #     train_ann = json.load(f)
        #     data.update([c['name'] for c in train_ann["categories"]])
        self.tao_to_coco_mapping = {int(v): int(k) for k, v in COCO_TO_TAO_MAPPING.items()}
        self.data = data
        self.image_files_root_path = Path(data["path"]) / 'frames'
        assert not (self.use_segments and self.use_keypoints), 'Can not use both segments and keypoints.'
        super().__init__(*args, **kwargs)
        # The paths of the images are loaded by BaseDataset class but in this case in a wrong way,
        # so we need to recreate the paths of the images using the labels
        self.recreate_im_file_paths_using_labels()
        assert len(self.labels) == len(self.im_files), 'Number of labels and images must match as they are loaded by index out of self.labels and self.im_files'
    
    def recreate_im_file_paths_using_labels(self):
        """Filter images by class in order to get same length of self.labels and self.im_files"""
        self.im_files = [str(self.image_files_root_path / label['im_file']) for label in self.labels]

    def get_img_files(self, img_path):
        """Read image files. Img path contains the .json file with the paths"""
        try:
            # Read image files using .json annotations
            with open(img_path, 'r') as f:
                dataset_info = json.load(f)
            im_files = [str(self.image_files_root_path / img['file_name']) for img in dataset_info['images']]

        except Exception as e:
            raise FileNotFoundError(f'{self.prefix}Error loading data from {img_path}\n{HELP_URL}') from e
        if self.fraction < 1:
            im_files = im_files[:round(len(im_files) * self.fraction)]
        return im_files

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
        """Builds and appends transforms to the list.
        Users can custom augmentations here
        like:
            if self.augment:
                # Training transforms
                return Compose([])
            else:
                # Val transforms
                return Compose([])
        """
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
            # transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(bbox_format='xywh',  # In reality it is "CXCYWH"
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask))
        return transforms

    def get_labels(self):
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)

        print(f'** Getting labels of {self.img_path} **')
        
        # Use pycocotools to load annotations
        print('Loading annotations into memory...')
        tic = time.time()
        with open(self.img_path, 'r') as f:
            dataset = json.load(f)
        assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
        print('Done (t={:0.2f}s)'.format(time.time()- tic))
        self.dataset = dataset
        self.createIndex()

        # Now properly load in the labels list
        print('Converting labels to Ultralytics format...')
        if self.data["coco_classes"] == True:
            print('Loading only COCO classes')
            list_of_coco_ids_in_tao = list(self.tao_to_coco_mapping.keys())
        t_init = time.perf_counter()
        cat_ids = []
        area = []
        labels = []
        for img_id in self.imgs:  # keys are img_ids
            current_img_boxes = []
            current_img_cls = []
            current_img_track_id = []
            assert img_id == self.imgs[img_id]['id'], "images are not sorted by id"
            ann_ids_for_img = self.getAnnIds(img_id, cat_ids, area)
            current_img_info = self.imgs[img_id]
            current_img_anns = [self.anns[ann_id] for ann_id in ann_ids_for_img]

            # If indicated in the YAML file, we only load the annotations of the COCO classes
            if self.data["coco_classes"]:
                for ann in current_img_anns:
                    if ann['category_id'] in list_of_coco_ids_in_tao:
                        current_img_boxes.append(ann['bbox'])
                        current_img_cls.append(self.tao_to_coco_mapping[ann['category_id']])
                        current_img_track_id.append(ann['track_id'])
            else:
                for ann in current_img_anns:
                    current_img_boxes.append(ann['bbox'])
                    current_img_cls.append(ann['category_id'])
                    current_img_track_id.append(ann['track_id'])

            
            if len(current_img_cls) > 0:
                # Classes
                classes = np.array(current_img_cls, dtype=np.float32)
                # Convert boxes from xywh to cxcywh and normalize bboxes
                h, w = (current_img_info['height'], current_img_info['width'])  # Convetion is HxW
                bboxes = np.array(current_img_boxes, dtype=np.float32)
                bboxes[:, 0] += bboxes[:, 2] / 2  # X from left corner to center
                bboxes[:, 1] += bboxes[:, 3] / 2  # Y from top corner to center
                bboxes[:, [0, 2]] /= w  # Normalize X
                bboxes[:, [1, 3]] /= h  # Normalize Y
                # Append to labels list
                labels.append(
                        {
                            'im_file': current_img_info['file_name'],
                            'shape': (h, w),  # Height x Width is the convention
                            'cls': classes[:, np.newaxis],  # [N, 1]
                            'bboxes': bboxes,  # [N, 4]
                            'segments': [],
                            'keypoints': None,
                            'normalized': True,  # In order to properly work, bboxes must be normalized
                            'bbox_format': 'xywh',  # "xywh" format is referred to "CXCYWH" format in this version of Ultralytics YOLO
                            'track_id': np.array(current_img_track_id, dtype=np.float32),
                        }
                )
            
        print(f'Done! (t={time.perf_counter() - t_init:0.2f}s)')
        return labels
    
    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

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
    
    ### Functions from cocoapi start here ###
    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats
    
    @staticmethod
    def _isArrayLike(obj):
        return hasattr(obj, '__iter__') and hasattr(obj, '__len__')
        
    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if self._isArrayLike(imgIds) else [imgIds]
        catIds = catIds if self._isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids
    ### Functions from cocoapi end here ###
    