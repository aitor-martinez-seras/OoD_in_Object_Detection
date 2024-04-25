from typing import List, Tuple, Union

import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, measure, io
from skimage.color import rgb2gray
from skimage.transform import resize
import matplotlib.patches as patches
from torch import Tensor

def extract_bboxes_from_saliency_map_and_thresholds(saliency_map: np.ndarray, thresholds: List[float]) -> List[np.ndarray]:
    """
    Extract bounding boxes from a saliency map using a list of thresholds to binarize the saliency map.
    Parameters:
        saliency_map: Tensor with shape (height, width). The saliency map to extract the bounding boxes.
        thresholds: List of float. The thresholds to binarize the saliency map.
    Returns:
        List of np.ndarray. Each position of the list are the thresholds obtained by each threshold. 
            Each element is a bounding box with shape (4,). The bounding box is represented as [minr, minc, maxr, maxc], 
            which corresponds to [y_min, x_min, y_max, x_max]. 
    """
    boxes_per_thr = []
    for i, thresh in enumerate(thresholds):
        boxes_one_thr = []
        binary_mask = saliency_map > thresh
        labeled_mask = measure.label(binary_mask)
        regions = measure.regionprops(labeled_mask)
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            x_min, y_min, x_max, y_max = minc, minr, maxc, maxr
            boxes_one_thr.append([x_min, y_min, x_max, y_max])
        # Convert to numpy array and append to the list per threshold
        boxes_per_thr.append(np.array(boxes_one_thr))
    return boxes_per_thr

def ftmap_minus_mean_of_ftmaps_then_abs_sum(ftmaps: np.ndarray) -> np.ndarray:
    """
    Compute the sum of the absolute values of the difference between
    the maximum value of each feature map and the mean of each feature map.
    Parameters:
        ftmaps: Tensor with shape (num_channels, height, width). 
                The feature maps of only one stride
    Returns:
        Tensor with shape (height, width)
    """
    mean_of_evey_ftmap = ftmaps.mean(axis=(1,2))
    ftmaps_minus_mean = ftmaps - mean_of_evey_ftmap[:, None, None]
    saliency_map = np.abs(ftmaps_minus_mean).sum(axis=0)
    return saliency_map

# Function to apply recursive Otsu thresholding
def recursive_otsu(image: np.ndarray, num_classes: int = 4, current_depth: int = 1, thresholds=None):
    """
    Apply recursive Otsu thresholding to an image.        
    """
    if thresholds is None:
        thresholds = []

    if current_depth < num_classes - 1:
        thresh = filters.threshold_otsu(image)
        thresholds.append(thresh)
        lower_region = image[image <= thresh]
        upper_region = image[image > thresh]

        recursive_otsu(lower_region, num_classes, current_depth + 1, thresholds)
        recursive_otsu(upper_region, num_classes, current_depth + 1, thresholds)

    return sorted(set(thresholds))
