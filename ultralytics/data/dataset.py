# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
from collections import defaultdict
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Union, List, Dict
import os
import glob
import re

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

from ultralytics.utils import LOCAL_RANK, LOGGER, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import resample_segments, segments2boxes
from ultralytics.utils.torch_utils import TORCHVISION_0_18

from .augment import (
    Compose,
    Format,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
)
from .base import BaseDataset
from .converter import merge_multi_segment
from .utils import (
    HELP_URL,
    check_file_speeds,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    verify_image_label,
)

# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
DATASET_CACHE_VERSION = "1.0.3"


class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    This class supports loading data for object detection, segmentation, pose estimation, and oriented bounding box
    (OBB) tasks using the YOLO format.

    Attributes:
        use_segments (bool): Indicates if segmentation masks should be used.
        use_keypoints (bool): Indicates if keypoints should be used for pose estimation.
        use_obb (bool): Indicates if oriented bounding boxes should be used.
        data (dict): Dataset configuration dictionary.

    Methods:
        cache_labels: Cache dataset labels, check images and read shapes.
        get_labels: Returns dictionary of labels for YOLO training.
        build_transforms: Builds and appends transforms to the list.
        close_mosaic: Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations.
        update_labels_info: Updates label format for different tasks.
        collate_fn: Collates data samples into batches.

    Examples:
        >>> dataset = YOLODataset(img_path="path/to/images", data={"names": {0: "person"}}, task="detect")
        >>> dataset.get_labels()
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """
        Initialize the YOLODataset.

        Args:
            data (dict, optional): Dataset configuration dictionary.
            task (str): Task type, one of 'detect', 'segment', 'pose', or 'obb'.
            *args (Any): Additional positional arguments for the parent class.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, channels=self.data["channels"], **kwargs)

    def cache_labels(self, path=Path("./labels.cache")):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file.

        Returns:
            (dict): Dictionary containing cached labels and related information.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                    repeat(self.single_cls),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        """
        Returns dictionary of labels for YOLO training.

        This method loads labels from disk or cache, verifies their integrity, and prepares them for training.

        Returns:
            (List[dict]): List of label dictionaries, each containing information about an image and its annotations.
        """
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"No images found in {cache_path}, training may not work correctly. {HELP_URL}")
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"No labels found in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def build_transforms(self, hyp=None):
        """
        Builds and appends transforms to the list.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms.
        """
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def close_mosaic(self, hyp):
        """
        Disable mosaic, copy_paste, mixup and cutmix augmentations by setting their probabilities to 0.0.

        Args:
            hyp (dict): Hyperparameters for transforms.
        """
        hyp.mosaic = 0.0
        hyp.copy_paste = 0.0
        hyp.mixup = 0.0
        hyp.cutmix = 0.0
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """
        Custom your label format here.

        Args:
            label (dict): Label dictionary containing bboxes, segments, keypoints, etc.

        Returns:
            (dict): Updated label dictionary with instances.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # make sure segments interpolate correctly if original length is greater than segment_resamples
            max_len = max(len(s) for s in segments)
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            # list[np.array(segment_resamples, 2)] * num_samples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """
        Collates data samples into batches.

        Args:
            batch (List[dict]): List of dictionaries containing sample data.

        Returns:
            (dict): Collated batch with stacked tensors.
        """
        new_batch = {}
        batch = [dict(sorted(b.items())) for b in batch]  # make sure the keys are in the same order
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img", "text_feats"}:
                value = torch.stack(value, 0)
            elif k == "visuals":
                value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


class YOLOMultiModalDataset(YOLODataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format with multi-modal support.

    This class extends YOLODataset to add text information for multi-modal model training, enabling models to
    process both image and text data.

    Methods:
        update_labels_info: Adds text information for multi-modal model training.
        build_transforms: Enhances data transformations with text augmentation.

    Examples:
        >>> dataset = YOLOMultiModalDataset(img_path="path/to/images", data={"names": {0: "person"}}, task="detect")
        >>> batch = next(iter(dataset))
        >>> print(batch.keys())  # Should include 'texts'
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """
        Initialize a YOLOMultiModalDataset.

        Args:
            data (dict, optional): Dataset configuration dictionary.
            task (str): Task type, one of 'detect', 'segment', 'pose', or 'obb'.
            *args (Any): Additional positional arguments for the parent class.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        super().__init__(*args, data=data, task=task, **kwargs)

    def update_labels_info(self, label):
        """
        Add texts information for multi-modal model training.

        Args:
            label (dict): Label dictionary containing bboxes, segments, keypoints, etc.

        Returns:
            (dict): Updated label dictionary with instances and texts.
        """
        labels = super().update_labels_info(label)
        # NOTE: some categories are concatenated with its synonyms by `/`.
        # NOTE: and `RandomLoadText` would randomly select one of them if there are multiple words.
        labels["texts"] = [v.split("/") for _, v in self.data["names"].items()]

        return labels

    def build_transforms(self, hyp=None):
        """
        Enhances data transformations with optional text augmentation for multi-modal training.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms including text augmentation if applicable.
        """
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            # NOTE: this implementation is different from official yoloe,
            # the strategy of selecting negative is restricted in one dataset,
            # while official pre-saved neg embeddings from all datasets at once.
            transform = RandomLoadText(
                max_samples=min(self.data["nc"], 80),
                padding=True,
                padding_value=self._get_neg_texts(self.category_freq),
            )
            transforms.insert(-1, transform)
        return transforms

    @property
    def category_names(self):
        """
        Return category names for the dataset.

        Returns:
            (Set[str]): List of class names.
        """
        names = self.data["names"].values()
        return {n.strip() for name in names for n in name.split("/")}  # category names

    @property
    def category_freq(self):
        """Return frequency of each category in the dataset."""
        texts = [v.split("/") for v in self.data["names"].values()]
        category_freq = defaultdict(int)
        for label in self.labels:
            for c in label["cls"].squeeze(-1):  # to check
                text = texts[int(c)]
                for t in text:
                    t = t.strip()
                    category_freq[t] += 1
        return category_freq

    @staticmethod
    def _get_neg_texts(category_freq, threshold=100):
        """Get negative text samples based on frequency threshold."""
        return [k for k, v in category_freq.items() if v >= threshold]


class GroundingDataset(YOLODataset):
    """
    Handles object detection tasks by loading annotations from a specified JSON file, supporting YOLO format.

    This dataset is designed for grounding tasks where annotations are provided in a JSON file rather than
    the standard YOLO format text files.

    Attributes:
        json_file (str): Path to the JSON file containing annotations.

    Methods:
        get_img_files: Returns empty list as image files are read in get_labels.
        get_labels: Loads annotations from a JSON file and prepares them for training.
        build_transforms: Configures augmentations for training with optional text loading.

    Examples:
        >>> dataset = GroundingDataset(img_path="path/to/images", json_file="annotations.json", task="detect")
        >>> len(dataset)  # Number of valid images with annotations
    """

    def __init__(self, *args, task="detect", json_file="", **kwargs):
        """
        Initialize a GroundingDataset for object detection.

        Args:
            json_file (str): Path to the JSON file containing annotations.
            task (str): Must be 'detect' or 'segment' for GroundingDataset.
            *args (Any): Additional positional arguments for the parent class.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        assert task in {"detect", "segment"}, "GroundingDataset currently only supports `detect` and `segment` tasks"
        self.json_file = json_file
        super().__init__(*args, task=task, data={"channels": 3}, **kwargs)

    def get_img_files(self, img_path):
        """
        The image files would be read in `get_labels` function, return empty list here.

        Args:
            img_path (str): Path to the directory containing images.

        Returns:
            (list): Empty list as image files are read in get_labels.
        """
        return []

    def verify_labels(self, labels):
        """Verify the number of instances in the dataset matches expected counts."""
        instance_count = sum(label["bboxes"].shape[0] for label in labels)
        if "final_mixed_train_no_coco_segm" in self.json_file:
            assert instance_count == 3662344
        elif "final_mixed_train_no_coco" in self.json_file:
            assert instance_count == 3681235
        elif "final_flickr_separateGT_train_segm" in self.json_file:
            assert instance_count == 638214
        elif "final_flickr_separateGT_train" in self.json_file:
            assert instance_count == 640704
        else:
            assert False

    def cache_labels(self, path=Path("./labels.cache")):
        """
        Loads annotations from a JSON file, filters, and normalizes bounding boxes for each image.

        Args:
            path (Path): Path where to save the cache file.

        Returns:
            (dict): Dictionary containing cached labels and related information.
        """
        x = {"labels": []}
        LOGGER.info("Loading annotation file...")
        with open(self.json_file) as f:
            annotations = json.load(f)
        images = {f"{x['id']:d}": x for x in annotations["images"]}
        img_to_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)
        for img_id, anns in TQDM(img_to_anns.items(), desc=f"Reading annotations {self.json_file}"):
            img = images[f"{img_id:d}"]
            h, w, f = img["height"], img["width"], img["file_name"]
            im_file = Path(self.img_path) / f
            if not im_file.exists():
                continue
            self.im_files.append(str(im_file))
            bboxes = []
            segments = []
            cat2id = {}
            texts = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= float(w)
                box[[1, 3]] /= float(h)
                if box[2] <= 0 or box[3] <= 0:
                    continue

                caption = img["caption"]
                cat_name = " ".join([caption[t[0] : t[1]] for t in ann["tokens_positive"]]).lower().strip()
                if not cat_name:
                    continue

                if cat_name not in cat2id:
                    cat2id[cat_name] = len(cat2id)
                    texts.append([cat_name])
                cls = cat2id[cat_name]  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                    if ann.get("segmentation") is not None:
                        if len(ann["segmentation"]) == 0:
                            segments.append(box)
                            continue
                        elif len(ann["segmentation"]) > 1:
                            s = merge_multi_segment(ann["segmentation"])
                            s = (np.concatenate(s, axis=0) / np.array([w, h], dtype=np.float32)).reshape(-1).tolist()
                        else:
                            s = [j for i in ann["segmentation"] for j in i]  # all segments concatenated
                            s = (
                                (np.array(s, dtype=np.float32).reshape(-1, 2) / np.array([w, h], dtype=np.float32))
                                .reshape(-1)
                                .tolist()
                            )
                        s = [cls] + s
                        segments.append(s)
            lb = np.array(bboxes, dtype=np.float32) if len(bboxes) else np.zeros((0, 5), dtype=np.float32)

            if segments:
                classes = np.array([x[0] for x in segments], dtype=np.float32)
                segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in segments]  # (cls, xy1...)
                lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
            lb = np.array(lb, dtype=np.float32)

            x["labels"].append(
                {
                    "im_file": im_file,
                    "shape": (h, w),
                    "cls": lb[:, 0:1],  # n, 1
                    "bboxes": lb[:, 1:],  # n, 4
                    "segments": segments,
                    "normalized": True,
                    "bbox_format": "xywh",
                    "texts": texts,
                }
            )
        x["hash"] = get_hash(self.json_file)
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        """
        Load labels from cache or generate them from JSON file.

        Returns:
            (List[dict]): List of label dictionaries, each containing information about an image and its annotations.
        """
        cache_path = Path(self.json_file).with_suffix(".cache")
        try:
            cache, _ = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.json_file)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, _ = self.cache_labels(cache_path), False  # run cache ops
        [cache.pop(k) for k in ("hash", "version")]  # remove items
        labels = cache["labels"]
        # self.verify_labels(labels)
        self.im_files = [str(label["im_file"]) for label in labels]
        if LOCAL_RANK in {-1, 0}:
            LOGGER.info(f"Load {self.json_file} from cache file {cache_path}")
        return labels

    def build_transforms(self, hyp=None):
        """
        Configures augmentations for training with optional text loading.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms including text augmentation if applicable.
        """
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            # NOTE: this implementation is different from official yoloe,
            # the strategy of selecting negative is restricted in one dataset,
            # while official pre-saved neg embeddings from all datasets at once.
            transform = RandomLoadText(
                max_samples=80,
                padding=True,
                padding_value=self._get_neg_texts(self.category_freq),
            )
            transforms.insert(-1, transform)
        return transforms

    @property
    def category_names(self):
        """Return unique category names from the dataset."""
        return {t.strip() for label in self.labels for text in label["texts"] for t in text}

    @property
    def category_freq(self):
        """Return frequency of each category in the dataset."""
        category_freq = defaultdict(int)
        for label in self.labels:
            for text in label["texts"]:
                for t in text:
                    t = t.strip()
                    category_freq[t] += 1
        return category_freq

    @staticmethod
    def _get_neg_texts(category_freq, threshold=100):
        """Get negative text samples based on frequency threshold."""
        return [k for k, v in category_freq.items() if v >= threshold]


class YOLOConcatDataset(ConcatDataset):
    """
    Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets for YOLO training, ensuring they use the same
    collation function.

    Methods:
        collate_fn: Static method that collates data samples into batches using YOLODataset's collation function.

    Examples:
        >>> dataset1 = YOLODataset(...)
        >>> dataset2 = YOLODataset(...)
        >>> combined_dataset = YOLOConcatDataset([dataset1, dataset2])
    """

    @staticmethod
    def collate_fn(batch):
        """
        Collates data samples into batches.

        Args:
            batch (List[dict]): List of dictionaries containing sample data.

        Returns:
            (dict): Collated batch with stacked tensors.
        """
        return YOLODataset.collate_fn(batch)

    def close_mosaic(self, hyp):
        """
        Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations.

        Args:
            hyp (dict): Hyperparameters for transforms.
        """
        for dataset in self.datasets:
            if not hasattr(dataset, "close_mosaic"):
                continue
            dataset.close_mosaic(hyp)


# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):
    """Semantic Segmentation Dataset."""

    def __init__(self):
        """Initialize a SemanticDataset object."""
        super().__init__()


class ClassificationDataset:
    """
    Extends torchvision ImageFolder to support YOLO classification tasks.

    This class offers functionalities like image augmentation, caching, and verification. It's designed to efficiently
    handle large datasets for training deep learning models, with optional image transformations and caching mechanisms
    to speed up training.

    Attributes:
        cache_ram (bool): Indicates if caching in RAM is enabled.
        cache_disk (bool): Indicates if caching on disk is enabled.
        samples (list): A list of tuples, each containing the path to an image, its class index, path to its .npy cache
                        file (if caching on disk), and optionally the loaded image array (if caching in RAM).
        torch_transforms (callable): PyTorch transforms to be applied to the images.
        root (str): Root directory of the dataset.
        prefix (str): Prefix for logging and cache filenames.

    Methods:
        __getitem__: Returns subset of data and targets corresponding to given indices.
        __len__: Returns the total number of samples in the dataset.
        verify_images: Verifies all images in dataset.
    """

    def __init__(self, root, args, augment=False, prefix=""):
        """
        Initialize YOLO object with root, image size, augmentations, and cache settings.

        Args:
            root (str): Path to the dataset directory where images are stored in a class-specific folder structure.
            args (Namespace): Configuration containing dataset-related settings such as image size, augmentation
                parameters, and cache settings.
            augment (bool, optional): Whether to apply augmentations to the dataset.
            prefix (str, optional): Prefix for logging and cache filenames, aiding in dataset identification.
        """
        import torchvision  # scope for faster 'import ultralytics'

        # Base class assigned as attribute rather than used as base class to allow for scoping slow torchvision import
        if TORCHVISION_0_18:  # 'allow_empty' argument first introduced in torchvision 0.18
            self.base = torchvision.datasets.ImageFolder(root=root, allow_empty=True)
        else:
            self.base = torchvision.datasets.ImageFolder(root=root)
        self.samples = self.base.samples
        self.root = self.base.root

        # Initialize attributes
        if augment and args.fraction < 1.0:  # reduce training fraction
            self.samples = self.samples[: round(len(self.samples) * args.fraction)]
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""
        self.cache_ram = args.cache is True or str(args.cache).lower() == "ram"  # cache images into RAM
        if self.cache_ram:
            LOGGER.warning(
                "Classification `cache_ram` training has known memory leak in "
                "https://github.com/ultralytics/ultralytics/issues/9824, setting `cache_ram=False`."
            )
            self.cache_ram = False
        self.cache_disk = str(args.cache).lower() == "disk"  # cache images on hard drive as uncompressed *.npy files
        self.samples = self.verify_images()  # filter out bad images
        self.samples = [list(x) + [Path(x[0]).with_suffix(".npy"), None] for x in self.samples]  # file, index, npy, im
        scale = (1.0 - args.scale, 1.0)  # (0.08, 1.0)
        self.torch_transforms = (
            classify_augmentations(
                size=args.imgsz,
                scale=scale,
                hflip=args.fliplr,
                vflip=args.flipud,
                erasing=args.erasing,
                auto_augment=args.auto_augment,
                hsv_h=args.hsv_h,
                hsv_s=args.hsv_s,
                hsv_v=args.hsv_v,
            )
            if augment
            else classify_transforms(size=args.imgsz)
        )

    def __getitem__(self, i):
        """
        Returns subset of data and targets corresponding to given indices.

        Args:
            i (int): Index of the sample to retrieve.

        Returns:
            (dict): Dictionary containing the image and its class index.
        """
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram:
            if im is None:  # Warning: two separate if statements required here, do not combine this with previous line
                im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        # Convert NumPy array to PIL image
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def verify_images(self):
        """
        Verify all images in dataset.

        Returns:
            (list): List of valid samples after verification.
        """
        desc = f"{self.prefix}Scanning {self.root}..."
        path = Path(self.root).with_suffix(".cache")  # *.cache file path

        try:
            check_file_speeds([file for (file, _) in self.samples[:5]], prefix=self.prefix)  # check image read speeds
            cache = load_dataset_cache_file(path)  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash([x[0] for x in self.samples])  # identical hash
            nf, nc, n, samples = cache.pop("results")  # found, missing, empty, corrupt, total
            if LOCAL_RANK in {-1, 0}:
                d = f"{desc} {nf} images, {nc} corrupt"
                TQDM(None, desc=d, total=n, initial=n)
                if cache["msgs"]:
                    LOGGER.info("\n".join(cache["msgs"]))  # display warnings
            return samples

        except (FileNotFoundError, AssertionError, AttributeError):
            # Run scan if *.cache retrieval failed
            nf, nc, msgs, samples, x = 0, 0, [], [], {}
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
                pbar = TQDM(results, desc=desc, total=len(self.samples))
                for sample, nf_f, nc_f, msg in pbar:
                    if nf_f:
                        samples.append(sample)
                    if msg:
                        msgs.append(msg)
                    nf += nf_f
                    nc += nc_f
                    pbar.desc = f"{desc} {nf} images, {nc} corrupt"
                pbar.close()
            if msgs:
                LOGGER.info("\n".join(msgs))
            x["hash"] = get_hash([x[0] for x in self.samples])
            x["results"] = nf, nc, len(samples), samples
            x["msgs"] = msgs  # warnings
            save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
            return samples


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
        # 1. Check if we are in OWOD style datasets or COCO OOD/Mixed datasets
        self.owod_task = kwargs['hyp'].get('owod_task', None)
        self.ood_or_mixed = self.data.get("ood_or_mixed", None)

        if self.ood_or_mixed:
            # Case COCO OOD or Mixed
            import json
            assert kwargs["hyp"].get("split") == 'val', 'COCO OOD and Mixed datasets are only available for the validation split'
            #assert self.owod_task is None, 'OWOD task is not available for COCO OOD and Mixed datasets'
            self.number_of_classes = 20  # Only 20 classes in COCO OOD or Mixed
            print(f"Using COCO {self.ood_or_mixed} dataset")

            # 2. Load the annotations from the json files
            if self.ood_or_mixed == 'ood':
                annotations_json_path = self.data["json_files"][0]
                with open(annotations_json_path, 'r') as f:
                    annotations = json.load(f)
            elif self.ood_or_mixed == 'mixed':
                annotations_ind_json_path = self.data["json_files"][0]
                annotations_ood_json_path = self.data["json_files"][1]
                # TODO: Aqui me las tengo que arreglar para convertir esto en un solo dict o algo asi
                with open(annotations_ind_json_path, 'r') as f:
                    annotations_ind = json.load(f)
                with open(annotations_ood_json_path, 'r') as f:
                    annotations_ood = json.load(f)
                # Merge the annotations
                annotations = annotations_ind
                for ann in annotations_ood["annotations"]:
                    annotations["annotations"].append(ann)
            else:
                raise ValueError(f'Invalid value for ood_or_mixed: {self.ood_or_mixed}')
            
            # 3. Create the labels using the annotations
            self.labels = self.create_labels_using_coco_ood_json_annotations(annotations)
            # 3.1: In case we want to use only a limited set of images
            from custom_hyperparams import CUSTOM_HYP
            if CUSTOM_HYP.USE_ONLY_SUBSET_OF_IMAGES:
                self.select_subset_of_images(images_to_select=CUSTOM_HYP.IMAGES_TO_SELECT)
            # 4: Update the attributes related with the labels (im_files, ni, npy_files, ...)
            self.update_attributes_to_new_labels()
            assert len(self.labels) == len(self.im_files), 'Number of labels and images must be equal'

        else:
            # Case OWOD or VOC or COCO standard
            if self.owod_task:
                print(f'Using OWOD task {self.owod_task} to filter dataset')
            else:
                print(f'Using the number of classes defined in the dataset to filter the dataset')
            self.number_of_classes = self.select_number_of_classes_owod(self.owod_task)
            self.number_of_classes = len(self.data["names"]) if self.number_of_classes == 0 else self.number_of_classes

            # 2: Update the labels
            self.upate_labels_to_use_less_classes()  # TODO: Separar esta clase en dos clases, una para OWOD y otra para limitar el numero de clases a traves de los names
            
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
    
    def select_subset_of_images(self, images_to_select: List[str]):
        """
        Select a subset of images to use in the dataset. This method is used to select a subset of images
        when we want to use only a limited set of images for training, validation or testing.
        """
        # Get the images that are part of the subset
        subset_images = [im_path + '.jpg' for im_path in images_to_select]
        # Filter the labels to include only the ones present in the subset
        self.labels = [lb for lb in self.labels if lb['im_file'].split('/')[-1] in subset_images]

    def upate_labels_to_use_less_classes(self):
        """
        This method maps the COCO classes to the OWOD classes if we are in OWOD mode and then
        filters the labels to include only the ones present in the YAML file in "names".        
        """
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

    ### COCO OOD and Mixed Methods ###
    def create_labels_using_coco_ood_json_annotations(self, annotations: Dict[str, List]) -> List[Dict]:
        """Replace the labels with the annotations from the json file."""
        # Take the images from the labels and create a dictionary with the image id as key and the image file as value
        img_files = {lb["im_file"].split('/')[-1]: lb["im_file"] for lb in self.labels}

        # Fill a dictionary with the annotations per image id
        annotations_per_image_id = {}
        for image in annotations["images"]:
            image_id = image["id"]
            annotations_per_image_id[image_id] = {
                "im_file": img_files[image["file_name"]],
                "shape": (image["height"], image["width"]),  # Height x Width is the convention
                "cls": [],  # Placeholder for the class
                "bboxes": [],  # Placeholder for the bbox
                "segments": [],
                "keypoints": None,  
                "bbox_format": "xywh",
                "normalized": True
            }
        set_img_ids_in_annotations_per_img = set(annotations_per_image_id.keys())

        # Using the annotations img id, fill the cls and bboxes iteratively
        print('WARNING: COCO OOD and Mixed classes, start at 1, so a -1 is applied to the cls to match OWOD classes.')
        coco_ood_to_owod_mapping = self.data.get('coco_ood_to_owod_mapping')
        img_ids_not_in_the_actual_images = set()
        for ann in annotations["annotations"]:
            image_id = ann["image_id"]

            # WARNING: This is because some image IDs of the annotations of the mixed_OOD are not in the 
            #   actual images (anntations["images"]), so I assume we skip them, as I assume in UnSniffer they do it
            if image_id not in set_img_ids_in_annotations_per_img:
                img_ids_not_in_the_actual_images.add(image_id)
                continue

            # Transform the cls to OWOD convention. First subtract 1 to match COCO classes, then apply mapping
            ann_cls = ann["category_id"] - 1  # COCO OOD and Mixed classes start at 1
            if ann_cls != 80:
                ann_cls = coco_ood_to_owod_mapping[ann_cls]
            annotations_per_image_id[image_id]["cls"].append([ann_cls])
            # Obtain height and width of the image to normalize the bbox
            im_height, im_width = annotations_per_image_id[image_id]["shape"]  # Height x Width is the convention
            # Transform to cx, cy, w, h normalized
            bbox = ann["bbox"]
            x, y, w, h = bbox  # Annotation of bboxes come in xywh format
            cx = (x + w / 2) / im_width
            cy = (y + h / 2) / im_height
            # Append bbox
            annotations_per_image_id[image_id]["bboxes"].append(
                [cx, cy, w / im_width, h / im_height]
            )
        print(f'WARNING: {len(img_ids_not_in_the_actual_images)} images in the annotations are not in the actual images, SKIPPING them!')
        
        # Finally, convert the dictionary to a list of labels and the cls and bboxes to numpy arrays
        labels = list(annotations_per_image_id.values())
        for label in labels:
            if len(label["cls"]) == 0:
                label["cls"] = np.empty((0, 1), dtype=np.float32)
                label["bboxes"] = np.empty((0, 4), dtype=np.float32)
            else:
                label["cls"] = np.array(label["cls"], dtype=np.float32)
                label["bboxes"] = np.array(label["bboxes"], dtype=np.float32)

        return labels

    ### OWOD Methods ###
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

    def read_img_files_from_txt(self, file_path):
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