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

COCO_TO_TAO_MAPPING = {"13": 91, "34": 58, "33": 621, "49": 747, "8": 118, "51": 221, "1": 95, "73": 126, "79": 1122, "27": 729, "48": 926, "61": 1117, "11": 1038, "40": 1215, "74": 276, "21": 78, "75": 1162, "68": 699, "55": 185, "47": 13, "59": 79, "30": 982, "60": 371, "65": 896, "14": 99, "63": 642, "6": 1135, "64": 717, "53": 829, "70": 1115, "67": 235, "0": 805, "32": 41, "10": 452, "25": 1155, "7": 1144, "43": 625, "35": 60, "23": 502, "4": 4, "12": 779, "57": 1001, "38": 1099, "24": 34, "46": 45, "45": 139, "36": 980, "39": 133, "16": 382, "29": 480, "50": 154, "20": 429, "2": 211, "54": 392, "28": 36, "41": 347, "78": 544, "37": 1057, "9": 1132, "62": 1097, "44": 1018, "17": 579, "3": 714, "22": 1229, "15": 229, "77": 1091, "26": 35, "71": 979, "66": 299, "5": 174, "42": 475, "56": 237, "72": 428, "76": 937, "18": 961, "58": 852, "31": 993, "19": 81}


class TAODataset(BaseDataset):

    def __init__(self, *args, data=None, use_segments=False, use_keypoints=False, **kwargs):
        self.use_segments = use_segments
        self.use_keypoints = use_keypoints
        # # TODO: Añadir la opcion de ponerle las clases aqui a partir de el .json de las annotations
        # with open(Path(data["path"]) / data["train"], 'r') as f:
        #     train_ann = json.load(f)
        #     data.update([c['name'] for c in train_ann["categories"]])
        self.data = data
        assert not (self.use_segments and self.use_keypoints), 'Can not use both segments and keypoints.'
        super().__init__(*args, **kwargs)

    def get_img_files(self, img_path):
        """Read image files. Img path contains the .json file with the paths"""
        try:
            images_folder_path = Path(self.data["path"]) / "frames"
            # Read image files using .json annotations
            with open(img_path, 'r') as f:
                dataset_info = json.load(f)
            im_files = [str(images_folder_path / img['file_name']) for img in dataset_info['images']]
            
            # for p in img_path if isinstance(img_path, list) else [img_path]:
            #     p = Path(p)  # os-agnostic
            #     if p.is_dir():  # dir
            #         f += glob.glob(str(p / '**' / '*.*'), recursive=True)
            #         # F = list(p.rglob('*.*'))  # pathlib
            #     elif p.is_file():  # file
            #         with open(p) as t:
            #             t = t.read().strip().splitlines()
            #             parent = str(p.parent) + os.sep
            #             f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
            #             # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
            #     else:
            #         raise FileNotFoundError(f'{self.prefix}{p} does not exist')
            # im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            # # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            # assert im_files, f'{self.prefix}No images found'
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
    
        # from torchvision.utils import draw_bounding_boxes
        # import matplotlib.pyplot as plt
        # from torchvision.ops import box_convert
        # im = draw_bounding_boxes(
        #                 torch.tensor(label['img']).permute(2,0,1),
        #                 box_convert(torch.tensor(label['bboxes']), 'xywh', 'xyxy'),
        #                 width=5,
        #                 font='FreeMonoBold',
        #                 font_size=12,
        #                 labels=[f'{self.data["names"][n]}' for i, n in enumerate(label['cls'])]
        #             )
        # fig,ax = plt.subplots(1,1,figsize=(20,10))
        # plt.imshow(im.permute(1,2,0))
        # plt.savefig('prueba_en_ood_visu.png')
        # plt.close()

        # # Plot image
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # ax.imshow(label['img'])
        # plt.savefig('prueba.png')
        # plt.close()

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
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(bbox_format='xywh',
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
        
        # TODO: Find a way to not hardcode the list of coco ids
        use_only_coco_classes = True
        
        with open('datasets_utils/tao/coco_id2tao_id.json', 'r') as f:
            # Keys are coco ids and values are tao ids
            coco_to_tao_mapping = json.load(f)
        tao_to_coco_mapping = {int(v): int(k) for k, v in coco_to_tao_mapping.items()}
        list_of_coco_ids_in_tao = list(tao_to_coco_mapping.keys())
        print(f'Getting labels of {self.img_path}')
        print('loading annotations into memory...')
        tic = time.time()
        with open(self.img_path, 'r') as f:
            dataset = json.load(f)
        assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
        print('Done (t={:0.2f}s)'.format(time.time()- tic))
        self.dataset = dataset
        self.createIndex()
        # We can modify the dataset when creating the object OAKDataset to only use 
        #   certain classes that 
        print('Converting labels to Ultralytics format...')
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

            # TODO: Filter to only COCO classes
            if use_only_coco_classes:
                for ann in current_img_anns:
                    if ann['category_id'] in list_of_coco_ids_in_tao:
                        current_img_boxes.append(ann['bbox'])
                        current_img_cls.append(tao_to_coco_mapping[ann['category_id']])
                        current_img_track_id.append(ann['track_id'])
            else:
                for ann in current_img_anns:
                    current_img_boxes.append(ann['bbox'])
                    current_img_cls.append(ann['category_id'])
                    current_img_track_id.append(ann['track_id'])

            if len(current_img_cls) > 0:
                labels.append(
                        {
                            'im_file': current_img_info['file_name'],
                            'shape': (current_img_info['height'], current_img_info['width']),
                            'cls': np.array(current_img_cls, dtype=np.float32),
                            'bboxes': np.array(current_img_boxes, dtype=np.float32),
                            'segments': [],
                            'keypoints': None,
                            'normalized': False,
                            'bbox_format': 'xywh',
                            'track_id': np.array(current_img_track_id),
                        }
                )
            
        print(f'Done! (t={time.perf_counter() - t_init:0.2f}s)')
        return labels
            
    
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
    