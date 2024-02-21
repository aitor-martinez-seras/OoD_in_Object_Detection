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
from ultralytics.yolo.data.augment import Format, LetterBox, Instances
from ultralytics.yolo.utils import DEFAULT_CFG


# classes in https://github.com/oakdata/benchmark/blob/fdb94230fc716efd6c96af355b106ec43ca64d08/object_detection/detectron2/otherfile/mapping.json

class TAODataset(BaseDataset):

    # IMAGE_SIZE = 1152x648

    def __init__(self,
                 imgs_path,
                 ann_path=None,
                 bboxes_format='xywh',
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
        self.bboxes_format = bboxes_format
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

    # # Transforms coco annotations to the format of Ultralytics
    # def get_labels(self):
    #     """Users can custom their own format here.
    #     Make sure your output is a list with each element like below:
    #         dict(
    #             im_file=im_file,
    #             shape=shape,  # format: (height, width)
    #             cls=cls,
    #             bboxes=bboxes, # xywh
    #             segments=segments,  # xy
    #             keypoints=keypoints, # xy
    #             normalized=True, # or False
    #             bbox_format="xyxy",  # or xywh, ltwh
    #         )
    #     """
    #     with open(self.ann_path, 'r') as f:
    #         anns = json.load(f)
    #     imgs_info_per_id = {}
    #     for i, ann in enumerate(anns['images']):
    #         imgs_info_per_id[ann['id']] = {
    #             "file_name": ann['file_name'],
    #             "height": ann['height'],
    #             "width": ann['width'],
    #         }

    #     labels = []
    #     boxes_per_image = []
    #     cls_per_image = []
    #     previous_image_id = 0
    #     for i, ann in enumerate(anns['annotations']):
    #         current_img_info = imgs_info_per_id[ann['image_id']]
    #         current_image_id = ann['image_id']
    #         # If we are in the same image, append the boxes to the boxes_per_image list
    #         if current_image_id == previous_image_id:
    #             boxes_per_image.append(ann['bbox'])
    #             cls_per_image.append(ann['category_id'])
    #         # Otherwise we append the boxes and all the info to the labels list
    #         else:
    #             if len(boxes.shape) == 1:
    #                 boxes = boxes[np.newaxis, :]
    #             labels.append(
    #                 {
    #                     'im_file': current_img_info['file_name'],
    #                     'shape': (current_img_info['height'], current_img_info['width']),
    #                     'cls': np.array(ann['category_id'], ndmin=1),
    #                     'bboxes': boxes,
    #                     'segments': ann['segmentation'],
    #                     'keypoints': None,
    #                     'normalized': False,
    #                     'bbox_format': 'xywh'
    #                 }
    #             )
            
    #         else:
    #             boxes = np.array(ann['bbox'], dtype=np.float64)
            
    #         previous_image_id = current_image_id            

    #     return labels
    
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

        if not self.ann_path == None:
            print('loading annotations into memory...')
            tic = time.time()
            with open(self.ann_path, 'r') as f:
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
                            'cls': np.array(current_img_cls, dtype=np.float64),
                            'bboxes': np.array(current_img_boxes, dtype=np.float64),
                            'segments': [],
                            'keypoints': None,
                            'normalized': False,
                            'bbox_format': 'xywh',
                            'track_id': np.array(current_img_track_id),
                        }
                )

            # Read the image and plot it
            # from PIL import Image
            # import numpy as np
            # import matplotlib.pyplot as plt
            # from torchvision.utils import draw_bounding_boxes
            # from torchvision.ops import box_convert

            # pil_img = Image.open(f"/home/tri110414/nfs_home/datasets/OAK/val/Raw/{current_img_info['file_name']}")
            # img = np.array(pil_img)
            # im = draw_bounding_boxes(
            #     torch.tensor(img).permute(2, 0, 1),
            #     box_convert(torch.tensor(current_img_boxes), 'xywh', 'xyxy'),
            #     width=2,
            #     font='FreeMonoBold',
            #     font_size=12,
            #     labels=[f'{self.dataset["categories"][current_img_cls[i]]["name"]}' for i in range(len(current_img_cls))],
            # )
            # fig, ax = plt.subplots(figsize=(20, 10))
            # plt.imshow(im.permute(1, 2, 0))
            # plt.savefig('prueba_dentro_del_get_labels.png')
            # plt.close()
            
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

if __name__ == "__main__":
    # generate_counts('train')
    # generate_counts('val')
    # generate_bar_plot('train')
    # generate_bar_plot('val')
    generate_coco_annotations('train')
    generate_coco_annotations('val')
    # OAKDataset(imgs_path='/home/tri110414/nfs_home/datasets/OAK/train/Raw', 
    #             ann_path='/home/tri110414/nfs_home/datasets/OAK/train/train_annotations_coco.json',
    #             imgsz=1152,)