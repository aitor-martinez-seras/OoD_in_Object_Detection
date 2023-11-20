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


def generate_counts(split):
    # Now we want to count the number of boxes per class
    if split == 'train':
        root_path = Path("/home/tri110414/nfs_home/datasets/OAK/train/Labels")
    elif split == 'val':
        root_path = Path("/home/tri110414/nfs_home/datasets/OAK/val/Labels")
    else:
        raise ValueError('split must be either train or val')
    
    classes_json_path = 'datasets_utils/oak/oak_classes.json'
    classes_dict = json.load(open(classes_json_path))

    # Create the classes count dictionary with the same keys as the classes dictionary
    classes_count = {}
    for key in classes_dict.keys():
        classes_count[key] = 0

    for i, folder in tqdm(enumerate(root_path.iterdir())):
        for json_file in folder.iterdir():
            with open(json_file) as f:
                data = json.load(f)
                for ann in data:
                    classes_count[ann['category']] += 1

    # Save the classes count dictionary
    with open(f'datasets_utils/oak/{split}_classes_count.json', 'w') as fp:
        json.dump(classes_count, fp, indent=4)


def generate_bar_plot(split):
    # Load json file
    if split == 'train':
        classes_count = json.load(open('datasets_utils/oak/train_classes_count.json'))
    elif split == 'val':
        classes_count = json.load(open('datasets_utils/oak/val_classes_count.json'))
    else:
        raise ValueError('split must be either train or val')

    # Sort the dictionary
    most_freq_classes_ordered = OrderedDict(sorted(classes_count.items(), key=lambda x: x[1], reverse=True))

    # Load coco classes
    coco_classes = json.load(open('datasets_utils/oak/coco_classes.json'))

    fig, ax = plt.subplots(figsize=(20, 10))
    plt.bar(most_freq_classes_ordered.keys(), most_freq_classes_ordered.values())
    if split == 'train':
        plt.ylim(0, 10000)
    elif split == 'val':
        plt.ylim(0, 1000)
    # Make visible the tick labels by rotating them 90 degrees
    plt.xticks(rotation=90)
    # Make the xticks be plotted in green if they are in the coco_classes 
    # and red otherwise
    for tick in ax.get_xticklabels():
        if tick.get_text() in coco_classes.values():
            tick.set_color('green')
        else:
            tick.set_color('red')
    # Put text in the mid right side of the plot to indicate that green are coco classes
    # and red are not coco classes
    plt.text(0.65, 0.50, 'Green: coco classes\nRed: not coco classes', style='italic', 
             bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10}, transform=ax.transAxes,
             fontsize=25, horizontalalignment='center', verticalalignment='center')
    plt.savefig(f'datasets_utils/oak/bar_plot_{split}.pdf', bbox_inches='tight')
    plt.close()


def generate_coco_annotations(split):
    # Load json file
    if split == 'train':
        root_path = Path("/home/tri110414/nfs_home/datasets/OAK/train/Labels")
    elif split == 'val':
        root_path = Path("/home/tri110414/nfs_home/datasets/OAK/val/Labels")
    else:
        raise ValueError('split must be either train or val')
    
    classes_json_path = 'datasets_utils/oak/oak_classes.json'
    classes_dict = json.load(open(classes_json_path))

    # Create the coco annotations dictionary
    coco_annotations = {
        "info": {
            "description": "OAK Dataset",
            "url": "",
            "version": "1.0",
            "year": 2021,
            "contributor": "Tri110414",
            "date_created": "2021/05/20"
        },
        "licenses": [
            {
                "url": "",
                "id": 1,
                "name": "Unknown"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Load the categories dict
    coco_classes = json.load(open('datasets_utils/oak/oak_classes.json'))
    coco_categories = []
    for key, value in coco_classes.items():
        coco_categories.append({
            "id": value,
            "name": key,
            "supercategory": "object"
        })
    coco_annotations['categories'] = coco_categories

    # Load the images and annotations
    image_id = 0
    video_id = -1
    video_fname_prev_iteration = ''
    annotation_id = 0
    img_height = 648
    img_width = 1152

    for idx_folder, folder in tqdm(enumerate(sorted(root_path.iterdir()))):

        for idx_json, json_file in enumerate(sorted(folder.iterdir())):
            
            if split == 'train':
                # Get the video name and check the video id
                video_fname_parts = json_file.stem.split('_')[:-1]
                video_fname = f'{video_fname_parts[0]}_{video_fname_parts[1]}_{video_fname_parts[2]}.mp4'
                # Check if it has changed, if it has changed, then we have a new video and we have to update the video_id
                if video_fname != video_fname_prev_iteration:
                    video_id += 1
                video_fname_prev_iteration = video_fname

            elif split == 'val':

                video_fname = folder.name + '.mp4'
                video_id = idx_folder

            else:
                raise ValueError('split must be either train or val')

            # Add the image
            coco_annotations['images'].append({
                "id": image_id,
                "video_id": video_id,
                "width": img_width,
                "height": img_height,
                "file_name": folder.name + '/' + json_file.stem + '.jpg',
                "video_file_name": video_fname,
                "license": 1,
                "flickr_url": "",
                "coco_url": "",                        
                "date_captured": "2021/05/20",
            })

            with open(json_file) as f:
                data = json.load(f)

                for ann in data:
                    bbox_width = ann['box2d']['x2'] - ann['box2d']['x1']
                    bbox_height = ann['box2d']['y2'] - ann['box2d']['y1']
                    coco_annotations['annotations'].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "video_id": video_id,
                        "category_id": ann['id'],
                        "segmentation": [],
                        "area": bbox_width * bbox_height,
                        # x1 and y1 are the top left corner of the bbox
                        "bbox": [ann['box2d']['x1'], ann['box2d']['y1'], bbox_width, bbox_height],
                        "iscrowd": 0,
                    })
                    # Every position in the list is an annotation
                    annotation_id += 1
                    
                    
                # Read the image and plot it
                # from PIL import Image
                # import numpy as np
                # import matplotlib.pyplot as plt
                # from torchvision.utils import draw_bounding_boxes
                # from torchvision.ops import box_convert

                # img_id = 0
                # pil_img = Image.open(f"/home/tri110414/nfs_home/datasets/OAK/train/Raw/{coco_annotations['images'][image_id]['file_name']}")
                # img = np.array(pil_img)
                # current_img_cls = [a['category_id'] for a in coco_annotations['annotations'] if a['image_id'] == image_id]
                # current_img_boxes = [a['bbox'] for a in coco_annotations['annotations'] if a['image_id'] == image_id]
                # im = draw_bounding_boxes(
                #     torch.tensor(img).permute(2, 0, 1),
                #     box_convert(torch.tensor(current_img_boxes), 'xywh', 'xyxy'),
                #     width=2,
                #     font='FreeMonoBold',
                #     font_size=12,
                #     labels=[f'{coco_annotations["categories"][current_img_cls[i]]["name"]}' for i in range(len(current_img_cls))],
                # )
                # fig, ax = plt.subplots(figsize=(20, 10))
                # plt.imshow(im.permute(1, 2, 0))
                # plt.savefig('prueba_dentro_del_create_coco_ann.png')
                # plt.close()

                # from PIL import Image
                # import numpy as np
                # import matplotlib.pyplot as plt

                # pil_img = Image.open(f"/home/tri110414/nfs_home/datasets/OAK/train/Raw/{folder.name + '/' + json_file.stem + '.jpg'}")

                # # Load annotations json
                # import json
                # with open('/home/tri110414/nfs_home/datasets/OAK/train/Labels/step_01125/20150316_133901_259_00738.json') as f:
                #     data = json.load(f)

                # box = 8
                # print(ann["category"])
                # img = np.array(pil_img)
                # plt.imshow(img)
                # plt.hlines(ann['box2d']['y1'], ann['box2d']['x1'], ann['box2d']['x2'], colors='r', linestyles='solid')
                # plt.hlines(ann['box2d']['y2'], ann['box2d']['x1'], ann['box2d']['x2'], colors='r', linestyles='solid')
                # plt.vlines(ann['box2d']['x1'], ann['box2d']['y1'], ann['box2d']['y2'], colors='r', linestyles='solid')
                # plt.vlines(ann['box2d']['x2'], ann['box2d']['y1'], ann['box2d']['y2'], colors='r', linestyles='solid')
                # plt.savefig('prueba.png')
                # plt.close()
                    

            # Every json file is an image
            image_id += 1   

    # Save the coco annotations
    with open(f'datasets_utils/oak/{split}_annotations_coco.json', 'w') as fp:
        json.dump(coco_annotations, fp, indent=4)


    



# classes in https://github.com/oakdata/benchmark/blob/fdb94230fc716efd6c96af355b106ec43ca64d08/object_detection/detectron2/otherfile/mapping.json

class OAKDataset(BaseDataset):

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
        cat_ids = []
        area = []
        labels = []
        for img_id in self.imgs:  # keys are img_ids
            current_img_boxes = []
            current_img_cls = []
            assert img_id == self.imgs[img_id]['id'], "images are not sorted by id"
            ann_ids_for_img = self.getAnnIds(img_id, cat_ids, area)
            current_img_info = self.imgs[img_id]
            current_img_anns = [self.anns[ann_id] for ann_id in ann_ids_for_img]
            for ann in current_img_anns:
                current_img_boxes.append(ann['bbox'])
                current_img_cls.append(ann['category_id'])
            labels.append(
                    {
                        'im_file': current_img_info['file_name'],
                        'shape': (current_img_info['height'], current_img_info['width']),
                        'cls': np.array(current_img_cls),
                        'bboxes': np.array(current_img_boxes),
                        'segments': [],
                        'keypoints': None,
                        'normalized': False,
                        'bbox_format': 'xywh'
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