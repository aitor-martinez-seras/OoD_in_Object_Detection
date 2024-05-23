from typing import List, Tuple, Union, Callable
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, measure, io
from skimage.color import rgb2gray
from skimage.transform import resize
import matplotlib.patches as patches
from torch import Tensor
from scipy.stats import median_abs_deviation
from custom_hyperparams import CUSTOM_HYP

NUM_THRS = CUSTOM_HYP.unk.NUM_THRESHOLDS + 1

def extract_bboxes_from_saliency_map_and_thresholds(saliency_map: np.ndarray, thresholds: List[float]) -> List[Tensor]:
    """
    Extract bounding boxes from a saliency map using a list of thresholds to binarize the saliency map.
    Parameters:
        saliency_map: Tensor with shape (height, width). The saliency map to extract the bounding boxes.
        thresholds: List of float. The thresholds to binarize the saliency map.
    Returns:
        List of Tensor. Each position of the list are the bounding boxes obtained by each threshold. 
            Each Tensor is of shape (N_thr, 4). Where N_thr is the number of bounding boxes obtained by the threshold.
            The bounding box is represented as [minr, minc, maxr, maxc], which corresponds to [y_min, x_min, y_max, x_max]. 
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
        boxes_per_thr.append(torch.tensor(boxes_one_thr))
    return boxes_per_thr

###############################
# Heatmap sumarization methods
###############################

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


def ftmap_minus_mean_of_ftmaps_then_sum(ftmaps: np.ndarray) -> np.ndarray:
    """
    Compute the sum of the difference between the maximum value of each feature map and the mean of each feature map.
    Parameters:
        ftmaps: Tensor with shape (num_channels, height, width). 
                The feature maps of only one stride
    Returns:
        Tensor with shape (height, width)
    """
    mean_of_evey_ftmap = ftmaps.mean(axis=(1,2))
    ftmaps_minus_mean = ftmaps - mean_of_evey_ftmap[:, None, None]
    saliency_map = ftmaps_minus_mean.sum(axis=0)
    return saliency_map


def sum_of_ftmaps(ftmaps: np.ndarray) -> np.ndarray:
    """
    Compute the sum of the feature maps.
    Parameters:
        ftmaps: Tensor with shape (num_channels, height, width). 
                The feature maps of only one stride
    Returns:
        Tensor with shape (height, width)
    """
    return ftmaps.sum(axis=0)


def std_of_ftmaps(ftmaps: np.ndarray) -> np.ndarray:
    """
    Compute the standard deviation of the feature maps.
    Parameters:
        ftmaps: Tensor with shape (num_channels, height, width). 
                The feature maps of only one stride
    Returns:
        Tensor with shape (height, width)
    """
    return ftmaps.std(axis=0)

def iqr_of_ftmaps(ftmaps: np.ndarray) -> np.ndarray:
    """
    Compute the interquartile range of the feature maps.
    Parameters:
        ftmaps: Tensor with shape (num_channels, height, width). 
                The feature maps of only one stride
    Returns:
        Tensor with shape (height, width)
    """
    q1 = np.percentile(ftmaps, 25, axis=0)
    q3 = np.percentile(ftmaps, 75, axis=0)
    iqr = q3 - q1
    return iqr


def mean_absolute_deviation_of_ftmaps(ftmaps: np.ndarray) -> np.ndarray:
    """
    Compute the mean absolute deviation of the feature maps.
    Parameters:
        ftmaps: Tensor with shape (num_channels, height, width). 
                The feature maps of only one stride
    Returns:
        Tensor with shape (height, width)
    """
    mean_of_evey_ftmap = ftmaps.mean(axis=(1,2))
    ftmaps_minus_mean = ftmaps - mean_of_evey_ftmap[:, None, None]
    mad = np.abs(ftmaps_minus_mean).mean(axis=0)
    return mad

# Median Absolute Deviation (using scipy)
def median_absolute_deviation_of_ftmaps(ftmaps: np.ndarray) -> np.ndarray:
    """
    Compute the median absolute deviation of the feature maps.
    Parameters:
        ftmaps: Tensor with shape (num_channels, height, width). 
                The feature maps of only one stride
    Returns:
        Tensor with shape (height, width)
    """
    mean_of_evey_ftmap = ftmaps.mean(axis=(1,2))
    ftmaps_minus_mean = ftmaps - mean_of_evey_ftmap[:, None, None]
    mad = median_abs_deviation(ftmaps_minus_mean, axis=0)
    return mad


def select_ftmaps_summarization_method(option: str) -> Callable:
    """
    Select the feature maps summarization method to apply to the feature maps.
    Parameters:
        option: str. The name of the summarization method to apply.
    Returns:
        Callable. The summarization method to apply.
    """
    if option == 'ftmap_minus_mean_of_ftmaps_then_abs_sum':
        return ftmap_minus_mean_of_ftmaps_then_abs_sum
    elif option == 'ftmap_minus_mean_of_ftmaps_then_sum':
        return ftmap_minus_mean_of_ftmaps_then_sum
    elif option == 'sum_of_ftmaps':
        return sum_of_ftmaps
    elif option == 'std_of_ftmaps':
        return std_of_ftmaps
    elif option == 'iqr_of_ftmaps':
        return iqr_of_ftmaps
    elif option == 'mean_absolute_deviation_of_ftmaps':
        return mean_absolute_deviation_of_ftmaps
    elif option == 'median_absolute_deviation_of_ftmaps':
        return median_absolute_deviation_of_ftmaps
    else:
        raise ValueError(f"Invalid summarization method: {option}. The available method is 'ftmap_minus_mean_of_ftmaps_then_abs_sum'.")
    

###############################
# Image thresholding methods
###############################
# They always use the default parameters

def recursive_otsu(image: np.ndarray, num_classes: int = NUM_THRS, current_depth: int = 1, thresholds=None) -> List[float]:
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


def multi_threshold_otsu(image: np.ndarray, num_classes: int = NUM_THRS) -> List[float]:
    # Multi-level Otsu's Thresholding
    thresholds = filters.threshold_multiotsu(image, classes=num_classes)
    return sorted(set(thresholds))


def k_means_thresholding(image: np.ndarray, num_clusters: int = NUM_THRS) -> List[float]:
    from sklearn.cluster import KMeans
    # # Flatten the image for clustering
    # pixels = image.reshape(-1, 1)
    # # Apply K-means clustering
    # kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(pixels)
    # cluster_centers = np.sort(kmeans.cluster_centers_.flatten())
    # # Thresholding based on cluster centers
    # thresholds = np.mean(np.vstack([cluster_centers[:-1], cluster_centers[1:]]), axis=0)

    # Flatten the image to a 1D array
    flat_image = image.flatten().reshape(-1, 1)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto').fit(flat_image)
    cluster_centers = sorted(kmeans.cluster_centers_.flatten())
    
    # Calculate thresholds as the midpoints between consecutive cluster centers
    thresholds = [(cluster_centers[i] + cluster_centers[i + 1]) / 2 for i in range(len(cluster_centers) - 1)]
    return sorted(set(thresholds))


def quantile_thresholding(image: np.ndarray, num_quantiles: int = NUM_THRS) -> List[float]:
    thresholds = np.quantile(image, np.linspace(0, 1, num_quantiles + 1)[1:-1])
    return sorted(set(thresholds))


def kittler_illingworth_threshold(image: np.ndarray) -> List[float]:
    """
    Apply Kittler-Illingworth thresholding to an image.
    """
    thresh = filters.threshold_minimum(image)
    return [thresh]


def select_thresholding_method(option: str) -> Callable:
    """
    Select the thresholding method to apply to the saliency map.
    Parameters:
        option: str. The name of the thresholding method to apply.
    Returns:
        Callable. The thresholding method to apply.
    """
    if option == 'recursive_otsu':
        return recursive_otsu
    elif option == 'multithreshold_otsu':
        return multi_threshold_otsu
    elif option == 'k_means':
        return k_means_thresholding
    elif option == 'quantile':
        return quantile_thresholding
    else:
        raise ValueError(f"Invalid thresholding method: {option}. The available methods are 'otsu' and 'recursive_otsu'.")
    