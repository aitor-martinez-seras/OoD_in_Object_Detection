# Ultralytics YOLO 🚀, AGPL-3.0 license

from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Union
import os
import glob
import re

import cv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from ..utils import LOCAL_RANK, NUM_THREADS, TQDM_BAR_FORMAT, is_dir_writeable
from .augment import Compose, Format, Instances, LetterBox, classify_albumentations, classify_transforms, v8_transforms
from .base import BaseDataset
from .utils import HELP_URL, LOGGER, get_hash, img2label_paths, verify_image_label, IMG_FORMATS


class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        use_segments (bool, optional): If True, segmentation masks are used as labels. Defaults to False.
        use_keypoints (bool, optional): If True, keypoints are used as labels. Defaults to False.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """
    cache_version = '1.0.2'  # dataset labels *.cache version, >= 1.0.0 for YOLOv8
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(self, *args, data=None, use_segments=False, use_keypoints=False, **kwargs):
        self.use_segments = use_segments
        self.use_keypoints = use_keypoints
        self.data = data
        assert not (self.use_segments and self.use_keypoints), 'Can not use both segments and keypoints.'
        super().__init__(*args, **kwargs)

    def cache_labels(self, path=Path('./labels.cache')):
        """Cache dataset labels, check images and read shapes.
        Args:
            path (Path): path where to save the cache file (default: Path('./labels.cache')).
        Returns:
            (dict): labels.
        """
        x = {'labels': []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f'{self.prefix}Scanning {path.parent / path.stem}...'
        total = len(self.im_files)
        nkpt, ndim = self.data.get('kpt_shape', (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in (2, 3)):
            raise ValueError("'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                             "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image_label,
                                iterable=zip(self.im_files, self.label_files, repeat(self.prefix),
                                             repeat(self.use_keypoints), repeat(len(self.data['names'])), repeat(nkpt),
                                             repeat(ndim)))
            pbar = tqdm(results, desc=desc, total=total, bar_format=TQDM_BAR_FORMAT)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x['labels'].append(
                        dict(
                            im_file=im_file,
                            shape=shape,
                            cls=lb[:, 0:1],  # n, 1
                            bboxes=lb[:, 1:],  # n, 4
                            segments=segments,
                            keypoints=keypoint,
                            normalized=True,
                            bbox_format='xywh'))
                if msg:
                    msgs.append(msg)
                pbar.desc = f'{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            pbar.close()

        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.im_files)
        x['results'] = nf, nm, ne, nc, len(self.im_files)
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        if is_dir_writeable(path.parent):
            if path.exists():
                path.unlink()  # remove *.cache file if exists
            np.save(str(path), x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{self.prefix}New cache created: {path}')
        else:
            LOGGER.warning(f'{self.prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.')
        return x

    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')
        try:
            import gc
            gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
            print(f'{self.prefix}Loading labels from {cache_path}')
            cache, exists = np.load(str(cache_path), allow_pickle=True).item(), True  # load dict
            print(f'{self.prefix}Loaded!')
            gc.enable()
            assert cache['version'] == self.cache_version  # matches current version
            assert cache['hash'] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in (-1, 0):
            d = f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            tqdm(None, desc=self.prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        if nf == 0:  # number of labels found
            raise FileNotFoundError(f'{self.prefix}No labels found in {cache_path}, can not start training. {HELP_URL}')

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels = cache['labels']
        self.im_files = [lb['im_file'] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb['cls']), len(lb['bboxes']), len(lb['segments'])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f'WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, '
                f'len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. '
                'To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.')
            for lb in labels:
                lb['segments'] = []
        if len_cls == 0:
            raise ValueError(f'All labels empty in {cache_path}, can not start training without labels. {HELP_URL}')
        return labels

    # TODO: use hyp config to set all these augmentations
    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
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

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

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


# Classification dataloaders -------------------------------------------------------------------------------------------
class ClassificationDataset(torchvision.datasets.ImageFolder):
    """
    YOLO Classification Dataset.

    Args:
        root (str): Dataset path.

    Attributes:
        cache_ram (bool): True if images should be cached in RAM, False otherwise.
        cache_disk (bool): True if images should be cached on disk, False otherwise.
        samples (list): List of samples containing file, index, npy, and im.
        torch_transforms (callable): torchvision transforms applied to the dataset.
        album_transforms (callable, optional): Albumentations transforms applied to the dataset if augment is True.
    """

    def __init__(self, root, args, augment=False, cache=False):
        """
        Initialize YOLO object with root, image size, augmentations, and cache settings.

        Args:
            root (str): Dataset path.
            args (Namespace): Argument parser containing dataset related settings.
            augment (bool, optional): True if dataset should be augmented, False otherwise. Defaults to False.
            cache (Union[bool, str], optional): Cache setting, can be True, False, 'ram' or 'disk'. Defaults to False.
        """
        super().__init__(root=root)
        if augment and args.fraction < 1.0:  # reduce training fraction
            self.samples = self.samples[:round(len(self.samples) * args.fraction)]
        self.cache_ram = cache is True or cache == 'ram'
        self.cache_disk = cache == 'disk'
        self.samples = [list(x) + [Path(x[0]).with_suffix('.npy'), None] for x in self.samples]  # file, index, npy, im
        self.torch_transforms = classify_transforms(args.imgsz)
        self.album_transforms = classify_albumentations(
            augment=augment,
            size=args.imgsz,
            scale=(1.0 - args.scale, 1.0),  # (0.08, 1.0)
            hflip=args.fliplr,
            vflip=args.flipud,
            hsv_h=args.hsv_h,  # HSV-Hue augmentation (fraction)
            hsv_s=args.hsv_s,  # HSV-Saturation augmentation (fraction)
            hsv_v=args.hsv_v,  # HSV-Value augmentation (fraction)
            mean=(0.0, 0.0, 0.0),  # IMAGENET_MEAN
            std=(1.0, 1.0, 1.0),  # IMAGENET_STD
            auto_aug=False) if augment else None

    def __getitem__(self, i):
        """Returns subset of data and targets corresponding to given indices."""
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram and im is None:
            im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f))
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        if self.album_transforms:
            sample = self.album_transforms(image=cv2.cvtColor(im, cv2.COLOR_BGR2RGB))['image']
        else:
            sample = self.torch_transforms(im)
        return {'img': sample, 'cls': j}

    def __len__(self) -> int:
        return len(self.samples)


# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):

    def __init__(self):
        """Initialize a SemanticDataset object."""
        super().__init__()


class FilteredYOLODataset(YOLODataset):
    """
    Dataset class for loading object detection labels in YOLO format with reduced number of classes
    with respect to the original dataset. This class must be defined in the corresponding YAML file 
    of the dataset. The number of classes to be used must be defined in the 'names' attribute of the dataset.
    In case of OWOD type datasets, the 'coco_to_owod_mapping' attribute must be defined in the dataset YAML file to 
    map the COCO classes to the OWOD classes. Also, the tasks .txt files must be defined in the 'owod' folder of the
    'datasets_utils' folder. Each .txt shows the images that are part of the task, being each line of the .txt the name
    of the image file without the extension and without the path.
    """

    def __init__(self, *args, data=None, use_segments=False, use_keypoints=False, **kwargs):
        super().__init__(*args, data=data, use_segments=use_segments, use_keypoints=use_keypoints, **kwargs)
        print(f' -- Filtering dataset --')
        # 1: Check if we are in OWOD style datasets and define the number of classes in consequence
        self.owod_task = kwargs['hyp'].get('owod_task', None)
        self.number_of_classes = self.select_number_of_classes_owod(self.owod_task)
        self.number_of_classes = len(self.data["names"]) if self.number_of_classes == 0 else self.number_of_classes
        # 2: Update the labels
        self.upate_labels_to_use_less_classes()
        # 3: Update the attributes related with the labels (im_files, ni, npy_files, ...)
        self.update_attributes_to_new_labels()
        # 4: Limit images to the ones that are part of the OWOD task (again checks if labels are correctly updated)
        if self.owod_task:
            self.limit_images_by_owod_tasks(self.owod_task)
        assert len(self.labels) == len(self.im_files), 'Number of labels and images must be equal'
        print(f'Succesfully filtered dataset. New dataset:')
        print(f'  * {len(self.labels)} images')
        print(f'  * {self.number_of_classes} classes')
        print(f'  * {sum(len(lb["cls"]) for lb in self.labels)} boxes')
        if self.owod_task:
            print(f'  * OWOD task: {self.owod_task}')
        print(f' -------------')
        
    ### Reimplementation of get_img_files in BaseDataset in ultralytics/yolo/data/base.py  ###
    # The only difference is that we are using the 'path' variable inside the .yaml file
    # of the dataset to define the root path for the images
    def get_img_files(self, img_path):
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        # New way of handling parent path selection
                        parent = self.data.get('path', '')
                        if parent:
                            parent = str(parent) + os.sep
                        else:
                            parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f'{self.prefix}{p} does not exist')
            im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f'{self.prefix}No images found'
        except Exception as e:
            raise FileNotFoundError(f'{self.prefix}Error loading data from {img_path}\n{HELP_URL}') from e
        if self.fraction < 1:
            im_files = im_files[:round(len(im_files) * self.fraction)]
        return im_files

    def upate_labels_to_use_less_classes(self):
        # 1: Map COCO classes to OWOD classes if we are in OWOD
        if self.data.get('coco_to_owod_mapping'):
            # If present, means we are in OWOD using Pascal class order
            # Therefore we need to update all class labels to the new mapping
            print(f'Updating labels to match the OWOD class order')
            self.map_coco_to_owod()
        # 2: Filter labels to include only the ones present in the YAML file in "names"
        classes = list(self.data["names"].keys())
        self.update_labels(include_class=classes)  # From BaseDataset
        if self.data.get('remove_images_with_no_annotations') is True:
            # Remove empty labels if indicated in the dataset YAML file
            print(f'Removing images with no annotations')
            self.remove_image_labels_with_no_annotations()
        else:
            print('Maintaining images with no annotations (background images)')

    def update_attributes_to_new_labels(self):        
        # Update image files related attributes
        self.im_files = [lb['im_file'] for lb in self.labels]  # update im_files
        self.ni = len(self.labels)
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
        # Update Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

    def remove_image_labels_with_no_annotations(self):
        """Remove images with no annotations."""
        self.labels = [lb for lb in self.labels if len(lb['cls']) > 0]

    def map_coco_to_owod(self):
        """Map COCO classes to OWOD classes."""
        mapping = self.data.get('coco_to_owod_mapping')
        coco_pattern = re.compile(r'^\d{12}\.jpg$')
        type_of_array = self.labels[0]['cls'].dtype
        for label in self.labels:
            #label['cls'] = np.array([[mapping[c[0]]] for c in label['cls']], dtype=type_of_array)
            # Check if 'cls' is empty
            if label['cls'].size == 0:
                # Directly create an empty array with the desired shape and type
                label['cls'] = np.empty((0, 1), dtype=type_of_array)
            else:
                # Proceed with mapping for non-empty arrays
                # and only if they are COCO images, as VOC labels are already in the correct order
                if coco_pattern.match(label["im_file"][-16:]):  # COCO images are always 12 digits long + .jpg
                    label['cls'] = np.array([[mapping[c[0]]] for c in label['cls']], dtype=type_of_array)
    
    def select_number_of_classes_owod(self, selected_owod_task: str) -> int:
        """Select the number of classes depending on the OWOD tasks."""
        if selected_owod_task == 't1':
            number_of_classes = 20
        elif selected_owod_task == 't2':
            number_of_classes = 40
        elif selected_owod_task == 't3':
            number_of_classes = 60
        elif selected_owod_task == 't4':
            number_of_classes = 80
        elif selected_owod_task == 'all_task_test':
            number_of_classes = 80
        else:
            number_of_classes = 0
        return number_of_classes

    def limit_images_by_owod_tasks(self, selected_owod_task: str):
        """Limit images to the ones that are part of the OWOD task."""
        # 1: Take the images that are part of the task
        img_files_to_include = self.retrieve_task_file_names(selected_owod_task)
        # Use python sets to improve lookup efficiency and use string operations rather than Path for the name
        set_of_included_imgs = set(img_files_to_include)
        self.labels = [lb for lb in self.labels if lb['im_file'].split('/')[-1].split('.')[0] in set_of_included_imgs]
        # 2: Check if any label that should not be in the dataset is still present
        classes = list(self.data["names"].keys())
        classes = classes[:self.number_of_classes]  # Filter classes to the task
        self.update_labels(include_class=classes)
        # 3: Update attributes
        self.update_attributes_to_new_labels()
        
    def retrieve_task_file_names(self, selected_owod_task: str) -> list[str]:
        """Retrieve the file names for the selected OWOD task."""
        root_path =  Path(__file__).resolve().parents[3]
        owod_tasks_path = root_path / 'datasets_utils' / 'owod' / 'tasks'
        mode = self._infer_mode()
        print(f'Using OWOD task {selected_owod_task} for {mode} mode')
        # Task 1
        if selected_owod_task == 't1':
            if mode == 'train':
                img_files_to_include = self.read_img_files_from_txt(owod_tasks_path / 't1_train.txt')
            elif mode == 'val':
                img_files_to_include = self.read_img_files_from_txt(owod_tasks_path / 't1_known_test.txt')
            else:
                raise ValueError(f'Invalid mode {mode}')
        # Task 2
        elif selected_owod_task == 't2':
            if mode == 'train':
                img_files_to_include = self.read_img_files_from_txt(owod_tasks_path / 't2_train.txt')
            elif mode == 'val':
                raise NotImplementedError('Validation set for task 2 is not available')
                img_files_to_include = self.read_img_files_from_txt(owod_tasks_path / 't2_val.txt')
            else:
                raise ValueError(f'Invalid mode {mode}')
        # Task 3
        elif selected_owod_task == 't3':
            if mode == 'train':
                img_files_to_include = self.read_img_files_from_txt(owod_tasks_path / 't3_train.txt')
            elif mode == 'val':
                raise NotImplementedError('Validation set for task 3 is not available')
                img_files_to_include = self.read_img_files_from_txt(owod_tasks_path / 't3_val.txt')
            else:
                raise ValueError(f'Invalid mode {mode}')
        # Task 4
        elif selected_owod_task == 't4':
            if mode == 'train':
                img_files_to_include = self.read_img_files_from_txt(owod_tasks_path / 't4_train.txt')
            elif mode == 'val':
                raise NotImplementedError('Validation set for task 4 is not available')
                img_files_to_include = self.read_img_files_from_txt(owod_tasks_path / 't4_val.txt')
            else:
                raise ValueError(f'Invalid mode {mode}')
        # All task test
        elif selected_owod_task == 'all_task_test':
            if mode == 'val':
                img_files_to_include = self.read_img_files_from_txt(owod_tasks_path / 'all_task_test.txt')
            else:
                raise ValueError(f'Invalid mode {mode} for task all_task_test')    
        
        else:
            raise ValueError(f'Invalid OWOD task selected: {selected_owod_task}')
        return img_files_to_include

    def read_img_files_from_txt(self, file_path: Union[str, Path]):
        """Read image files from a txt file."""
        with open(file_path, 'r') as f:
            return [line.rstrip() for line in f]

    def _infer_mode(self) -> str:
        img_files_paths_filename = Path(self.img_path).name
        if 'train' in img_files_paths_filename:
            return 'train'
        elif 'val' in img_files_paths_filename:
            return 'val'
        elif 'test' in img_files_paths_filename:
            return 'test'
        else:
            raise ValueError(f'Invalid mode for the file {img_files_paths_filename}')


    """ INFO ABOUT THE TASKS and SPLITS
     -- Task all_task_test --
    Contained in Val split

    -- Task all_task_val --
    Contained in Train split

    -- Task t1_known_test --
    Contained in Val split

    -- Task t1_train --
    Contained in Train split

    -- Task t1_train_with_unk --
    Contained in Train split

    -- Task t2_ft --
    Contained in Train split

    -- Task t2_train --
    Contained in Train split

    -- Task t2_train_with_unk --
    Contained in Train split

    -- Task t3_ft --
    Contained in Train split

    -- Task t3_train --
    Contained in Train split

    -- Task t3_train_with_unk --
    Contained in Train split

    -- Task t4_ft --
    Contained in Train split

    -- Task t4_train --
    Contained in Train split

    -- Task wr1 --
    Intersection with Train split: 4951/9903
    Intersection with Val split: 4952/9903
    Intersection with Test split: 0/9903
    """