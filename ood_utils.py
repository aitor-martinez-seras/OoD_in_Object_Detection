from typing import List, Tuple, Callable, Type, Union, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
from logging import Logger
import json 

import numpy as np
# import sklearn.metrics as sk
from sklearn.metrics import pairwise_distances
import torch
from torch import Tensor
import torchvision.ops as t_ops
from scipy.optimize import linear_sum_assignment

from ultralytics import YOLO
from ultralytics.yolo.data.build import InfiniteDataLoader
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.v8.detect.predict import extract_roi_aligned_features_from_correct_stride
from visualization_utils import plot_results
from datasets_utils.owod.owod_evaluation_protocol import compute_metrics
from constants import IND_INFO_CREATION_OPTIONS, AVAILABLE_CLUSTERING_METHODS, \
    AVAILABLE_CLUSTER_OPTIMIZATION_METRICS, INTERNAL_ACTIVATIONS_EXTRACTION_OPTIONS, \
    FTMAPS_RELATED_OPTIONS, LOGITS_RELATED_OPTIONS, OOD_METHOD_CHOICES, TARGETS_RELATED_OPTIONS


# Funciones para asignar peso a los feature maps
def weighted_variance(saliency_maps):
    # VALORES ALTOS DE ESTE NOS DICE QUE MAPAS TIENE PUNTOS MUY CONCENTRADOS CON ALTO VALOR, SIENDO EL RESTO
    # PRACTIAMENTE CERO
    # No tiene gran utilidad aparente
    weights = np.sum(saliency_maps, axis=(1, 2), keepdims=True)
    indices = np.indices(saliency_maps.shape[1:])
    mean_x = np.sum(indices[0] * saliency_maps, axis=(1, 2)) / weights.squeeze()
    mean_y = np.sum(indices[1] * saliency_maps, axis=(1, 2)) / weights.squeeze()
    variance_x = np.sum(saliency_maps * (indices[0] - mean_x[:, None, None])**2, axis=(1, 2)) / weights.squeeze()
    variance_y = np.sum(saliency_maps * (indices[1] - mean_y[:, None, None])**2, axis=(1, 2)) / weights.squeeze()
    return variance_x + variance_y

def spatial_frequency_analysis(saliency_maps):
    # Parece ser bueno para encontrar mapas utiles, pero tienes que usar:
    # topK_indices = np.argsort(spatial_frequency_analysis(ftmaps))[::-1][-topK:]
    # Es decir, hay que coger los de MENOR valor.
    freq_maps = np.fft.fft2(saliency_maps, axes=(1, 2))
    magnitude = np.abs(freq_maps)
    frequencies1 = np.fft.fftfreq(n=saliency_maps.shape[1], d=1.0)
    frequencies2 = np.fft.fftfreq(n=saliency_maps.shape[2], d=1.0)
    high_freq_power = np.sum(magnitude * (frequencies1[:, None]**2 + frequencies2[None, :]**2), axis=(1, 2))
    return high_freq_power

def center_of_mass_and_spread(saliency_maps):
    total_saliency = np.sum(saliency_maps, axis=(1, 2))
    coordinates = np.indices((saliency_maps.shape[1], saliency_maps.shape[2]))
    com_x = np.sum(coordinates[0] * saliency_maps, axis=(1, 2)) / total_saliency
    com_y = np.sum(coordinates[1] * saliency_maps, axis=(1, 2)) / total_saliency
    spread = np.sqrt(np.sum(saliency_maps * ((coordinates[0] - com_x[:, None, None])**2 + (coordinates[1] - com_y[:, None, None])**2), axis=(1, 2)) / total_saliency)
    return np.stack((com_x, com_y), axis=1), spread

def entropy(saliency_maps):
    # Puede ser util, ya que cuanta mas alta la entropia parace contener mas informacion
    # PROBLEMA: Hay NaNs... ¿Por que? ¿Como los manejamos?
    normalized_maps = saliency_maps / (np.sum(saliency_maps, axis=(1, 2), keepdims=True) + 1e-9)  # Avoid division by zero
    entropy = -np.sum(normalized_maps * np.log2(normalized_maps + 1e-9), axis=(1, 2))  # Avoid log(0)
    return entropy

import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, measure, io
from skimage.color import rgb2gray
from skimage.transform import resize
import matplotlib.patches as patches
# Function to apply recursive Otsu thresholding
def recursive_otsu(image, num_classes, current_depth=1, thresholds=None):
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

def multi_threshold_otsu(image, num_classes):
    # Multi-level Otsu's Thresholding
    thresholds = filters.threshold_multiotsu(image, classes=num_classes)
    return thresholds

def k_means_thresholding(image, num_clusters):
    from sklearn.cluster import KMeans
    # Flatten the image for clustering
    pixels = image.reshape(-1, 1)
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pixels)
    cluster_centers = np.sort(kmeans.cluster_centers_.flatten())
    # Thresholding based on cluster centers
    thresholds = np.mean(np.vstack([cluster_centers[:-1], cluster_centers[1:]]), axis=0)
    return thresholds


def draw_bounding_boxes_from_regions(regions, ax, color='blue', padding=None, stride=8, resized=False):
    # TODO: Añadir opcion de plot de OOD decision
    min_side_length = 3
    if resized:
        min_side_length = min_side_length * stride
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        # Remove small patches
        if (maxr - minr) >= min_side_length and (maxc - minc) >= min_side_length:  # Minimo 3 de lado
            #print(f"minr: {minr}, minc: {minc}, maxr: {maxr}, maxc: {maxc}")
            if padding is not None:
                minr += padding[0]
                minc += padding[1]
                maxr += padding[0]
                maxc += padding[1]
                minr *= stride
                minc *= stride
                maxr *= stride
                maxc *= stride
            rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                    fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(rect)

def basic_plots_for_one_set_of_thrs(saliency_map, thresholds, folder_path, original_image, padding, stride, method_name, ftmaps=None, ood_method=None):
    # Plotting thresholded images with bounding boxes for each threshold
    figsize = (len(thresholds)*4, 6)
    fig, axs = plt.subplots(1, len(thresholds), figsize=figsize)
    for i, thresh in enumerate(thresholds):
        binary_mask = saliency_map > thresh
        labeled_mask = measure.label(binary_mask)
        regions = measure.regionprops(labeled_mask)
        ax = axs[i]
        ax.imshow(binary_mask, cmap='hot')
        ax.set_title(f'Threshold > {thresh:.2f}')
        ood_decision = None  # TODO
        # Draw bounding boxes
        draw_bounding_boxes_from_regions(regions, ax, color='blue', ood_decision=ood_decision)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(folder_path / f"thresholds_with_bboxes_{method_name}.pdf")
    plt.close()

    # Plotting saliency with bounding boxes for each threshold
    fig, axs = plt.subplots(1, len(thresholds), figsize=figsize)
    for i, thresh in enumerate(thresholds):
        binary_mask = saliency_map > thresh
        labeled_mask = measure.label(binary_mask)
        regions = measure.regionprops(labeled_mask)
        ax = axs[i]
        ax.imshow(saliency_map, cmap='hot')
        ax.set_title(f'Threshold > {thresh:.2f}')
        ood_decision = None
        # Draw bounding boxes
        draw_bounding_boxes_from_regions(regions, ax, color='blue', ood_decision=ood_decision)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(folder_path / f"saliency_maps_with_boxes_separated_{method_name}.pdf")
    plt.close()

    # Plot the boxes for each threshold on the original image separately
    fig, axs = plt.subplots(1, len(thresholds), figsize=(12, 6))
    for i, thresh in enumerate(thresholds):
        binary_mask = saliency_map > thresh
        labeled_mask = measure.label(binary_mask)
        regions = measure.regionprops(labeled_mask)
        ax = axs[i]
        ax.imshow(original_image)
        ax.set_title(f'Threshold > {thresh:.2f}')
        ood_decision = None
        # Draw bounding boxes
        draw_bounding_boxes_from_regions(regions, ax, color='blue', padding=padding, stride=stride)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(folder_path / f"bboxes_on_original_separated_{method_name}.pdf")
    plt.close()
    

def generate_image_with_bboxes(saliency_map, folder_path, original_image, padding, option, ftmaps=None, ood_method=None):

    # Apply recursive Otsu thresholding to find multiple thresholds
    num_classes = 4  # For example, dividing the image into 4 classes
    thresholds = recursive_otsu(saliency_map, num_classes)
    # Figsize for the plots with multiple subplots (when thresholds are used)
    figsize = (len(thresholds)*4, 6)

    # Define the stride
    stride = 8
    if option == "various_thr_methods":
        # recursive_otsu
        basic_plots_for_one_set_of_thrs(saliency_map, thresholds, folder_path, original_image, padding, stride, f"recursive_otsu_{num_classes}_classes")

        # Multi-level Otsu's Thresholding
        thresholds = multi_threshold_otsu(saliency_map, num_classes)
        basic_plots_for_one_set_of_thrs(saliency_map, thresholds, folder_path, original_image, padding, stride, f"multi_level_otsu_{num_classes}_classes")

        # K-means clustering
        thresholds = k_means_thresholding(saliency_map, num_classes)
        basic_plots_for_one_set_of_thrs(saliency_map, thresholds, folder_path, original_image, padding, stride, f"k_means_clustering_{num_classes}_clusters")

        # Quartile thresholding
        thresholds = [np.quantile(saliency_map, 0.25), np.quantile(saliency_map, 0.5), np.quantile(saliency_map, 0.75), np.quantile(saliency_map, 0.85), np.quantile(saliency_map, 0.95)]
        basic_plots_for_one_set_of_thrs(saliency_map, thresholds, folder_path, original_image, padding, stride, "quartile_thresholding")
    
    elif option == "all_images":

        # Boxes on ORIGINAL SIZED FEATURE MAP
        # Use the thresholds to create binary masks
        fig, axs = plt.subplots(1, len(thresholds), figsize=figsize)
        for i, thresh in enumerate(thresholds):
            mask = saliency_map > thresh
            axs[i].imshow(mask, cmap='gray')
            axs[i].set_title(f'Threshold > {thresh:.2f}')
            axs[i].axis('off')
        plt.tight_layout()
        plt.savefig(folder_path / f"thresholds.pdf")
        plt.close()

        # Plotting thresholded images with bounding boxes for each threshold
        fig, axs = plt.subplots(1, len(thresholds), figsize=figsize)
        for i, thresh in enumerate(thresholds):
            binary_mask = saliency_map > thresh
            labeled_mask = measure.label(binary_mask)
            regions = measure.regionprops(labeled_mask)
            ax = axs[i]
            ax.imshow(binary_mask, cmap='hot')
            ax.set_title(f'Threshold > {thresh:.2f}')
            # Draw bounding boxes
            draw_bounding_boxes_from_regions(regions, ax, color='blue')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(folder_path / f"thresholds_with_bboxes.pdf")
        plt.close()

        # Boxes on RESHAPED SALIENCY MAP
        # Resized the saliency map to the original image size
        resized_saliency_map = resize(saliency_map, (saliency_map.shape[0]*8, saliency_map.shape[1]*8))
        resized_thresholds = recursive_otsu(resized_saliency_map, num_classes)
        fig, axs = plt.subplots(1, len(resized_thresholds), figsize=figsize)
        for i, thresh in enumerate(resized_thresholds):
            mask = resized_saliency_map > thresh
            axs[i].imshow(mask, cmap='gray')
            axs[i].set_title(f'Threshold > {thresh:.2f}')
            axs[i].axis('off')
        plt.tight_layout()
        plt.savefig(folder_path / f"resized_thresholds.pdf")
        plt.close()

        # Plotting saliency with bounding boxes for each threshold
        fig, axs = plt.subplots(1, len(resized_thresholds), figsize=figsize)
        for i, thresh in enumerate(resized_thresholds):
            binary_mask = resized_saliency_map > thresh
            labeled_mask = measure.label(binary_mask)
            regions = measure.regionprops(labeled_mask)

            ax = axs[i]
            ax.imshow(resized_saliency_map, cmap='hot')
            ax.set_title(f'Threshold > {thresh:.2f}')

            # Draw bounding boxes
            draw_bounding_boxes_from_regions(regions, ax, color='blue')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(folder_path / f"resized_bboxes.pdf")
        plt.close()

        # Plot all boxes in the original image taking into account that the boxes are generated
        # based on a reduced and non-padded image
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(original_image)
        for i, thresh in enumerate(thresholds):
            binary_mask = saliency_map > thresh
            labeled_mask = measure.label(binary_mask)
            regions = measure.regionprops(labeled_mask)

            # Draw bounding boxes
            draw_bounding_boxes_from_regions(regions, ax, color='blue', padding=padding, stride=stride)
        ax.axis('off')
        plt.savefig(folder_path / f"bboxes_on_original.pdf")
        plt.close()

        # Plot the boxes for each threshold on the original image separately
        fig, axs = plt.subplots(1, len(thresholds), figsize=(12, 6))
        for i, thresh in enumerate(thresholds):
            binary_mask = saliency_map > thresh
            labeled_mask = measure.label(binary_mask)
            regions = measure.regionprops(labeled_mask)
            ax = axs[i]
            ax.imshow(original_image)
            ax.set_title(f'Threshold > {thresh:.2f}')

            # Draw bounding boxes
            draw_bounding_boxes_from_regions(regions, ax, color='blue', padding=padding, stride=stride)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(folder_path / f"bboxes_on_original_separated.pdf")
        plt.close()

        # Plot the original image with saliency map on top (transparent)
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(original_image)
        # First add padding to the saliency map and then resize it to the original image size
        padded_saliency_map = np.pad(saliency_map, ((padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=(0, 0))
        resized_saliency_map = resize(padded_saliency_map, (original_image.shape[1], original_image.shape[0]))
        ax.imshow(resized_saliency_map, cmap='hot', alpha=0.5)
        ax.axis('off')
        plt.savefig(folder_path / f"original_with_saliency_map.pdf")
        plt.close()

        # Now the same but with also all the boxes on top
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(original_image)
        ax.imshow(resized_saliency_map, cmap='hot', alpha=0.5)
        # Draw bounding boxes
        for i, thresh in enumerate(thresholds):
            binary_mask = saliency_map > thresh
            labeled_mask = measure.label(binary_mask)
            regions = measure.regionprops(labeled_mask)

            # Draw bounding boxes
            draw_bounding_boxes_from_regions(regions, ax, color='blue', padding=padding, stride=stride)
        ax.axis('off')
        plt.savefig(folder_path / f"original_with_saliency_map_and_bboxes.pdf")
        plt.close()

        # Now the same but with also all the boxes on top
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(original_image)
        ax.imshow(resized_saliency_map, cmap='hot', alpha=0.5)
        # Draw bounding boxes
        for i, thresh in enumerate(thresholds):
            binary_mask = saliency_map > thresh
            labeled_mask = measure.label(binary_mask)
            regions = measure.regionprops(labeled_mask)

            # Draw bounding boxes
            draw_bounding_boxes_from_regions(regions, ax, color='blue', padding=padding, stride=stride)
        ax.axis('off')
        plt.savefig(folder_path / f"original_with_saliency_map_and_bboxes.pdf")
        plt.close()

    else:
        raise ValueError("Option not recognized")


#@dataclass(slots=True)
class OODMethod(ABC):
    """
    Base class for all the OOD methods. It contains the basic structure of the methods and the abstract methods that
    need to be overriden by each method. It also contains some helper functions that can be used by all the methods.
    Attributes:
        name: str -> Name of the method
        distance_method: bool -> True if the method uses a distance to measure the OOD, False if it uses a similarity
        per_class: bool -> True if the method computes a threshold for each class, False if it computes a single threshold
        per_stride: bool -> True if the method computes a threshold for each stride, False if it computes a single threshold
        thresholds: Union[List[float], List[List[float]]] -> The thresholds for each class and stride
        iou_threshold_for_matching: float -> The threshold to use when matching the predicted boxes to the ground truth boxes
        min_conf_threshold: float -> Define the minimum threshold to output a box when predicting
        which_internal_activations: str -> Where to extract internal activations from.
    """
    name: str
    distance_method: bool
    per_class: bool
    per_stride: bool
    thresholds: Union[List[float], List[List[float]]]
    # The threshold to use when matching the predicted boxes to the ground truth boxes.
    #   All boxes with an IoU lower than this threshold will be considered bad predictions
    iou_threshold_for_matching: float
    # Define the minimum threshold to output a box when predicting. All boxe with
    #   a confidence lower than this threshold will be discarded automatically
    min_conf_threshold: float
    which_internal_activations: str  # Where to extract internal activations from
    enhanced_unk_localization: bool  # If True, the method will try to enhance the localization of the UNK objects by adding new boxes

    def __init__(self, name: str, distance_method: bool, per_class: bool, per_stride: bool, iou_threshold_for_matching: float,
                 min_conf_threshold: float, which_internal_activations: str, enhanced_unk_localization: bool = False):
        self.name = name
        self.distance_method = distance_method
        self.per_class = per_class
        self.per_stride = per_stride
        self.iou_threshold_for_matching = iou_threshold_for_matching
        self.min_conf_threshold = min_conf_threshold
        self.thresholds = None  # This will be computed later
        self.which_internal_activations = self.validate_internal_activations_option(which_internal_activations)
        self.enhanced_unk_localization = enhanced_unk_localization

    def validate_internal_activations_option(self, selected_option: str):
        assert selected_option in INTERNAL_ACTIVATIONS_EXTRACTION_OPTIONS, f"Invalid option selected ({selected_option}) for " \
         f"internal activations extraction. Options are: {INTERNAL_ACTIVATIONS_EXTRACTION_OPTIONS}"
        return selected_option

    @abstractmethod
    def extract_internal_activations(self, results: Results, all_activations: Union[List[float], List[List[np.ndarray]]], targets: Dict[str, Tensor]):
        """
        Function to be overriden by each method to extract the internal activations of the model. In the logits
        methods, it will be the logits, and in the ftmaps methods, it will be the ftmaps.
        The extracted activations will be stored in the list all_activations
        """
        pass

    @abstractmethod
    def format_internal_activations(self, all_activations: Union[List[float], List[List[np.ndarray]]]):
        """
        Function to be overriden by each method to format the internal activations of the model.
        The extracted activations will be stored in the list all_activations
        """
        pass

    @abstractmethod
    def compute_ood_decision_on_results(self, results: Results, logger) -> List[List[int]]:
        """
        Function to be overriden by each method type to compute the OOD decision for each image.
        Parameters:
            results: Results -> The results of the model predictions
            logger: Logger -> The logger to print warnings or info
        Returns:
            ood_decision: List[int] -> A list of lists, where the first list is for each image and the second list is for each bbox. 
                The value is 1 if the bbox is In-Distribution, 0 if it is Out-of-Distribution
        """
        pass

    @abstractmethod
    def compute_score_one_bbox() -> List[float]:
        """
        Function to be overriden by each method to compute the score of one bbox.
        The input parameters will depend on if the method is a logits method or a distance method.
        """
        pass
    
    # TODO: Aqui vamos a computar las metricas AUROC, AUPR y FPR95
    def iterate_data_to_compute_ood_decision():
        pass
    
    @staticmethod
    def log_every_n_batches(n: int, logger, idx_of_batch: int, number_of_batches: int):
        if idx_of_batch % n == 0:
            logger.info(f"{(idx_of_batch/number_of_batches) * 100:02.1f}%: Procesing batch {idx_of_batch} of {number_of_batches}")
    
    @staticmethod
    def create_targets_dict(data: Dict) -> Dict[str, List[Tensor]]:
        """
        Function that creates a dictionary with the targets of each image in the batch with the following format:
            targets = {
                'bboxes': List[Tensor], each position refers to an image and each tensor has shape (n_bboxes, 4) and contains the bboxes in xyxy format
                'cls': List[Tensor], each position refers to an image and each tensor has shape (n_bboxes) and contains the classes of the bboxes
            }
        The bboxes are converted from relative to absolute coordinates and from cxcywh to xyxy format
        The function is necessary because the targets come in a single dimension, not separated by image,
        so we have to match each bbox to the corresponding image.
        """
        # Como los targets vienen en una sola dimension, tenemos que hacer el matcheo de a que imagen pertenece cada bbox
        # Para ello, tenemos que sacar en una lista, donde cada posicion se refiere a cada imagen del batch, los indices
        # de los targets que pertenecen a esa imagen
        target_idx = [torch.where(data['batch_idx'] == img_idx) for img_idx in range(len(data['im_file']))]
        # Tambien tenemos que sacar el tamaño de la imagen original para poder pasar de coordenadas relativas a absolutas
        # necesitamos que sea un array para poder luego indexar todas las bboxes de una al crear el dict targets
        relative_to_absolute_coordinates = [np.array(data['resized_shape'][img_idx] + data['resized_shape'][img_idx]) for img_idx in range(len(data['im_file']))]

        # Ahora creamos los targets
        # Targets es un diccionario con dos listas, con tantas posiciones como imagenes en el batch
        # En 'bboxes' tiene las bboxes. Hay que convertirlas a xyxy y pasar a coordenadas absolutas
        # En 'cls' tiene las clases
        # NOTE: Las bboxes viene en formato cxcywh, a pesar de que se llame xywh, ya que en esta version de Ultralytics es asi
        targets = dict(
            bboxes=[t_ops.box_convert(data['bboxes'][idx], 'cxcywh', 'xyxy') * relative_to_absolute_coordinates[img_idx] for img_idx, idx in enumerate(target_idx)],
            #bboxes=[data['bboxes'][idx] * relative_to_absolute_coordinates[img_idx] for img_idx, idx in enumerate(target_idx)],
            cls=[data['cls'][idx].view(-1) for idx in target_idx]    
        )
        return targets

    @staticmethod
    def match_predicted_boxes_to_targets(results: Results, targets, iou_threshold: float):
        """
        Function that creates the valid_preds list in the Results object. This list contains the indexes of the valid predictions.
        A valid prediction is a prediction that has been matched to a ground truth box with an IoU higher than the threshold.
        It uses the IoU and the class to match the boxes. If several boxes overlap with the same GT objetct and have 
        the same class as the object, the one with the highest IoU is selected.
        The results are stored in the Results object as follows:
            - assignment_score_matrix: The IoU matrix multiplied by the mask matrix
            - assignment: The result of the linear assignment
            - valid_preds: The list of the indexes of the valid predictions
        The final and most useful result is in the valid_preds list, where we have the indexes of the valid predictions.
        """
        # TODO: Optimize
        # Tenemos que conseguir matchear cada prediccion a una de las cajas Ground Truth (GT)
        for img_idx, res in enumerate(results):
            # Para ello, primero calculamos el IoU de cada caja predicha con cada caja GT
            # Despues creamos una matrix de misma shape que la de IoU, donde nos dice 
            # para cada caja predicha si la clase coincide con la de la correspondiente caja GT
            iou_matrix = t_ops.box_iou(res.boxes.xyxy.cpu(), targets['bboxes'][img_idx].cpu())
            mask_matrix = torch.zeros(res.boxes.cls.size()[0], len(targets['cls'][img_idx])) ## Tensor de zeros para rellenar con True False si las clases coinciden
            for i_pred in range(0,res.boxes.cls.size()[0]):
                for i_real in range(0, len(targets['cls'][img_idx])):
                    if res.boxes.cls[i_pred].cpu() == targets['cls'][img_idx][i_real]:
                        mask_matrix[i_pred, i_real] = True
                    else:
                        mask_matrix[i_pred, i_real] = False
            
            # Al multiplicarlas elemento a elemento, nos quedamos con los IoU de las cajas que coinciden en clase
            results[img_idx].assignment_score_matrix = iou_matrix * mask_matrix
            
            # Con el linear assigment podemos asignar a cada caja predicha la caja GT que mejor se ajusta
            results[img_idx].assignment = linear_sum_assignment(results[img_idx].assignment_score_matrix, maximize=True)
            
            # Finalmente, recorremos la segunda tupla de la asignacion, que nos dice a que caja GT se ha 
            # asignado cada caja predicha. Si el IoU es mayor que un threshold, la caja se considera correcta
            # y se añade a la lista de cajas validas. Las cajas que hayan sido asignadas a pesar de que 
            # su coste en la assignment_score_matrix sea 0, seran eliminadas al usar el threshold
            results[img_idx].valid_preds = []
            for i, assigment in enumerate(results[img_idx].assignment[1]):
                if results[img_idx].assignment_score_matrix[i, assigment] > iou_threshold:
                    results[img_idx].valid_preds.append(i)

    def iterate_data_to_extract_ind_activations(self, data_loader, model, device, logger):
        """
        Function to iterate over the data and extract the internal activations of the model for each image.
        The extracted activations will be stored in a list.
        """
        if self.per_class:
            if self.per_stride:
                all_internal_activations = [[[] for _ in range(3)] for _ in range(len(model.names))]
            else:
                all_internal_activations = [[] for _ in range(len(model.names))]

        # Obtain the bbox format from the last transform of the dataset
        if hasattr(data_loader.dataset.transforms.transforms[-1], "bbox_format"):
            bbox_format = data_loader.dataset.transforms.transforms[-1].bbox_format
        else:
            bbox_format=data_loader.dataset.labels[0]['bbox_format']

        # Start iterating over the data
        number_of_batches = len(data_loader)
        for idx_of_batch, data in enumerate(data_loader):
            
            self.log_every_n_batches(50, logger, idx_of_batch, number_of_batches)
                
            ### Prepare images and targets to feed the model ###
            imgs, targets = self.prepare_data_for_model(data, device)

            ### Process the images to get the results (bboxes and clasification) and the extra info for OOD detection ###
            results = model.predict(imgs, save=False, verbose=False)

            ### Match the predicted boxes to the ground truth boxes ###
            self.match_predicted_boxes_to_targets(results, targets, self.iou_threshold_for_matching)

            ### Extract the internal activations of the model depending on the OOD method ###
            self.extract_internal_activations(results, all_internal_activations, targets)

        ### Final formatting of the internal activations ###
        self.format_internal_activations(all_internal_activations)

        return all_internal_activations

    def prepare_data_for_model(self, data, device) -> Tuple[Tensor, dict]:
        """
        Funcion que prepara los datos para poder meterlos en el modelo.
        """
        if isinstance(data, dict):
            imgs = data['img'].to(device)
            targets = self.create_targets_dict(data)
        else:
            imgs, targets = data
        return imgs, targets

    def iterate_data_to_plot_with_ood_labels(self, model: YOLO, dataloader: InfiniteDataLoader, device: str, logger: Logger, folder_path: Path, now: str):
        
        # # Obtain the bbox format from the last transform of the dataset
        # if hasattr(dataloader.dataset.transforms.transforms[-1], "bbox_format"):
        #     bbox_format = dataloader.dataset.transforms.transforms[-1].bbox_format
        # else:
        #     bbox_format=dataloader.dataset.labels[0]['bbox_format']

        logger.warning(f"Using a confidence threshold of {self.min_conf_threshold} for tests")
        count_of_images = 0
        number_of_images_saved = 0

        # TODO: Estamos usando esta funcion para meter las diferentes pruebas... Finalmente deberia quedar solo
        #   la ejecucion normal. Al final lo que tenemos que dejar es el metodo que pone en el if como "normal"

        # TODO: MODOS PARA DEBUGEAR
        # localizacion: buscamos crear un algoritmo capaz de mejorar la localizacion de las cajas
        # clasificacion: queremos ver si localizando las cajas seriamos capaces de clasificarlas como UNK
        # normal: ejecucion normal, se predicen cajas y se ve si son UNK o no
        debugeando_en_modo = "normal"

        # # TODO: Activado para ejecutar los plots OOD con targets
        # if callable(getattr(self, "compute_ood_decision_with_ftmaps", None)):
        #     model.model.extraction_mode = 'all_ftmaps'

        c = 0
        for idx_of_batch, data in enumerate(dataloader):
            count_of_images += len(data['im_file'])
            # if count_of_images < 8000:
            #     continue
            ### Preparar imagenes y targets ###
            imgs, targets = self.prepare_data_for_model(data, device)
            
            ### Procesar imagenes en el modelo para obtener logits y las cajas ###
            results = model.predict(imgs, save=False, verbose=True, conf=self.min_conf_threshold)
            
            # TODO: Activado para ejecutar los plots OOD con targets
            if debugeando_en_modo == "clasificacion":
                ### Experiment with targets ###
                # Lo que hago es representar los targets para cada imagen con el label de Known o UNK en funcion de si son clases
                #   conocidas o no. Ademas, para cada uno de estos casos el score se ha obtenido de forma diferente. En ambos casos 
                #   Hay 3 valores, uno por cada stride, con el orden s8/s16/s32.
                # Known: Cada valor representa si para el espacio del stride y la clase correspondientes al objeto en la imagen, la caja es OoD o no
                # UNK: Cada valor representa el % de espacios de clase para el stride correspondiente que han considerado el objeto OOD
                if callable(getattr(self, "compute_ood_decision_with_ftmaps", None)):
                    # Create a list of lists with the roi aligned features of each bouding box of each image
                    # First list is for the image, second list is for the bbox

                    # 1.Group ftmaps of all images in the batch per level, so that 
                    #   we can do the RoIAlign efficiently once per level
                    ftmaps_per_level = []
                    for idx_lvl in range(3):
                        one_level_ftmaps = []
                        for i_bt in range(len(results)):
                            one_level_ftmaps.append(results[i_bt].extra_item[idx_lvl].to('cpu'))
                        ftmaps_per_level.append(torch.stack(one_level_ftmaps))
                    
                    # 2.Prepare the boxes in a list of boxes and keep track of the image index
                    #   so that we can match the boxes with the correct image. Each position in the list of bboxes
                    #   refers to one image bboxes
                    _bboxes = []
                    idx_of_img_per_box = []
                    for _idx, one_img_bboxes in enumerate(targets["bboxes"]):
                        _bboxes.append(one_img_bboxes.to(torch.float32))
                        idx_of_img_per_box.append(Tensor([_idx]*len(one_img_bboxes), dtype=torch.int32))
                    #_bboxes = torch.cat(_boxes, dim=0)
                    idx_of_img_per_box = torch.cat(idx_of_img_per_box, dim=0)
                    #_bboxes = [b.to(torch.float32) for b in targets["bboxes"]]
                    
                    # 3. RoIAlign the features per stride
                    roi_aligned_features_all_images_per_stride = []
                    
                    for idx_lvl in range(3):                        
                        roi_aligned_ftmaps_one_stride = t_ops.roi_align(
                            input=ftmaps_per_level[idx_lvl],
                            boxes=_bboxes,
                            output_size=(1,1),  # Tiene que ser el mismo que hayamos usado para entrenar el algoritmo OOD
                            spatial_scale=ftmaps_per_level[idx_lvl].shape[2]/imgs[0].shape[2]
                        )

                        roi_aligned_features_all_images_per_stride.append(roi_aligned_ftmaps_one_stride)

                    # Now create a list of lists, where the first list refers to the image and the second list to the stride
                    activations = []
                    for idx_img in range(len(results)):
                        positions_of_activations_to_select = torch.where(idx_of_img_per_box == idx_img)[0]
                        activations.append([])
                        for idx_stride in range(3):
                            activations[idx_img].append(roi_aligned_features_all_images_per_stride[idx_stride][positions_of_activations_to_select].numpy())

                    # OOD Decision is a list of lists, where the first list refers to the image and the second list to the bbox
                    # Each bbox has 3 scores, one per stride
                    ood_decision = self.compute_ood_decision_with_ftmaps(activations, bboxes=targets, logger=logger)

                    # PLOT #
                    width = 2
                    font = 'FreeMonoBold'
                    font_size = 11
                    # Create folder to store images
                    prueba_ahora_path = folder_path / f'{now}_{self.name}'
                    prueba_ahora_path.mkdir(exist_ok=True)

                    import matplotlib.pyplot as plt
                    from torchvision.utils import draw_bounding_boxes

                    for idx_img, bboxes_one_img in enumerate(targets['bboxes']):
                        
                        # The labels are the ood decisions for each bbox
                        labels_str = []
                        for idx_bbox in range(len(bboxes_one_img)):
                            if targets['cls'][idx_img][idx_bbox] <= 19:
                                # Use the name of the class
                                known_or_unk = f'{model.names[int(targets["cls"][idx_img][idx_bbox])]}'
                            else:
                                known_or_unk = 'UNK'
                            labels_str.append(f'{known_or_unk}: {ood_decision[idx_img][idx_bbox][0]}/{ood_decision[idx_img][idx_bbox][1]}/{ood_decision[idx_img][idx_bbox][2]}')

                        # The color will be red if the mean of the ood decisions is more than 0.5, green otherwise
                        colors = ['red' if sum(ood_decision[idx_img][idx_bbox]) > 1.5 else 'green' for idx_bbox in range(len(bboxes_one_img))]

                        im = draw_bounding_boxes(
                            imgs[idx_img].cpu(),
                            bboxes_one_img,
                            width=width,
                            font=font,
                            font_size=font_size,
                            labels=labels_str,
                            colors=colors
                        )

                        plt.imshow(im.permute(1,2,0))
                        plt.savefig(prueba_ahora_path / f'{(count_of_images + idx_img):03}.pdf', dpi=300)
                        plt.close()


                else:
                    raise NotImplementedError("The method does not have the function compute_ood_decision_with_ftmaps implemented")

            elif debugeando_en_modo == "localizacion":
                ### 1º prueba ###
                import matplotlib.pyplot as plt
                from torchvision.utils import draw_bounding_boxes
                from skimage.transform import resize
                from sklearn.metrics import pairwise_distances
                from sklearn import metrics
                from sklearn.cluster import DBSCAN
                import scipy.stats as sc_stats
                # Pintar los feature maps de algunas imagenes
                # Cogemos solo los mapas 80x80 de momento
                # Create folder to store images
                prueba_ahora_path = folder_path / f'{now}_{self.name}'
                prueba_ahora_path.mkdir(exist_ok=True)

                torch.set_printoptions(precision=2, threshold=10)
                np.set_printoptions(precision=2, threshold=10)

                # RELLENAR LISTA CON LAS VISUALIZACIONES QUE QUERAMOS
                modos_de_visualizacion = [
                    #"multiples_metricas",
                    #"subplots",
                    #"clusters",
                    #"ftmaps",
                    "bboxes",
                    ]

                one_ch_cmap = 'gray'
                for _img_idx, res in enumerate(results):
                    c += 1
                    if _img_idx == -1:
                        continue
                    else:
                        # Create folder for the image using the batch index and image index
                        ftmaps_path = prueba_ahora_path / f'{number_of_images_saved + _img_idx}'
                        ftmaps_path.mkdir(exist_ok=True)

                        # Obtain the original image shape from the data dict
                        # and figure out which zones of the feature maps 
                        # should be removed
                        original_shape = data['ori_shape'][_img_idx]
                        # The ratio pad has in its first dimension the factor to which the image has been 
                        # resized in each dimension (H, W) and
                        # in the second dimension the padding in each dimension (X, Y). It is important to note
                        # that the padding indicated is done by each of the sides of the image (left and right, top and bottom)
                        ratio_pad = np.array(data['ratio_pad'][_img_idx][1], dtype=float)  # Take the padding
                        # Remove the padding from the feature maps, but first we have to take into account that the
                        # ftmaps have a lower resolution than the original image. In this case, the ratio is 8.
                        ratio_pad_for_ftmaps = ratio_pad / 8  # For stride 8
                        x_padding = int(ratio_pad_for_ftmaps[0])  # The padding in the x dimension is the first element
                        y_padding = int(ratio_pad_for_ftmaps[1])  # The padding in the y dimension is the second element
                        ftmaps = res.extra_item[0].cpu().detach().numpy()
                        ftmap_height = ftmaps.shape[1]
                        ftmap_width = ftmaps.shape[2]
                        ftmaps = ftmaps[:, y_padding:ftmap_height-y_padding, x_padding:ftmap_width-x_padding]

                        # CREATE ONLY THE SUBPLOTS
                        if "subplots" in modos_de_visualizacion:
                            # Save also the original with annotations
                            orig_annotated = draw_bounding_boxes(
                                imgs[_img_idx].cpu(),
                                targets['bboxes'][_img_idx],
                                width=2,
                                font='FreeMonoBold',
                                font_size=11,
                                # Convert the classes to the names and above 19 all should be named unk
                                labels=[model.names[int(cls)] if cls <= 19 else "UNK" for cls in targets['cls'][_img_idx]],                        
                                colors=['blue']*len(targets['cls'][_img_idx])
                            )
                            # Mean
                            mean_ftmap = ftmaps.mean(axis=0)
                            mean_ftmap = (mean_ftmap - mean_ftmap.min()) / (mean_ftmap.max() - mean_ftmap.min())
                            # Add the padding
                            mean_ftmap = np.pad(mean_ftmap, ((y_padding, y_padding), (x_padding, x_padding)), 'constant', constant_values=(0, 0))
                            # STD
                            std_ftmap = ftmaps.std(axis=0)
                            std_ftmap = (std_ftmap - std_ftmap.min()) / (std_ftmap.max() - std_ftmap.min())
                            # Add the padding
                            std_ftmap = np.pad(std_ftmap, ((y_padding, y_padding), (x_padding, x_padding)), 'constant', constant_values=(0, 0))
                            # CLUSTERS. Clustering with DBSCAN but adding the spatial information
                            # Original feature map shape
                            num_channels, height, width = ftmaps.shape
                            # Create the coordinate channels
                            y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
                            # Add the coordinate channels to the original feature map
                            yx_augmented_feature_map = np.concatenate([ftmaps, y_coords[None, :, :], x_coords[None, :, :]])
                            # Now flatten the feature maps and make the first dimension be the pixels
                            flattened_ftmaps_with_yx = yx_augmented_feature_map.reshape(num_channels + 2, -1).T
                            ft_metric = 'cosine'
                            samples_pairwise_distances_features = pairwise_distances(flattened_ftmaps_with_yx[:, :-2], metric=ft_metric)
                            mean_ft_distance = samples_pairwise_distances_features.mean()
                            std_ft_distance = samples_pairwise_distances_features.std()
                            # Euclidean distance between sample coordinates. Normalize them to be in the same range as the feature distances
                            samples_pairwise_distances_yx = pairwise_distances(flattened_ftmaps_with_yx[:, -2:], metric='euclidean')
                            samples_pairwise_distances_yx = (samples_pairwise_distances_yx - samples_pairwise_distances_yx.min()) / (samples_pairwise_distances_yx.max() - samples_pairwise_distances_yx.min())
                            samples_pairwise_distances_yx = samples_pairwise_distances_yx * (std_ft_distance + mean_ft_distance - (mean_ft_distance - std_ft_distance)) + (mean_ft_distance - std_ft_distance)
                            # Merge the two distances using a weight lambda
                            lambda_weight = 0.5
                            samples_pairwise_distances_weighted = lambda_weight * samples_pairwise_distances_features + (1 - lambda_weight) * samples_pairwise_distances_yx
                            # Now cluster
                            eps = 18 if ft_metric == 'euclidean' else 0.13
                            db = DBSCAN(eps=eps, min_samples=16).fit(samples_pairwise_distances_weighted)
                            labels = db.labels_
                            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                            print(f"Number of clusters: {n_clusters_}")
                            # Plot the labels as an image in the original size
                            labels_as_image = labels.reshape(ftmaps.shape[1], ftmaps.shape[2])
                            # Add padding to the labels
                            labels_as_image = np.pad(labels_as_image, ((y_padding, y_padding), (x_padding, x_padding)), 'constant', constant_values=(-1, -1))
                            labels_as_image_spatial_info = labels_as_image.copy()
                            # Create a PNG of good quality with 4 subplots, the original image, the mean, the std and the clustering
                            # with no axis and no white space
                            fig, axs = plt.subplots(2, 2, figsize=(8, 8))
                            axs[0, 0].imshow(orig_annotated.permute(1,2,0))
                            axs[0, 0].axis('off')
                            axs[0, 0].set_title('Original (annotated)')
                            axs[0, 1].imshow(mean_ftmap, cmap=one_ch_cmap)
                            axs[0, 1].axis('off')
                            axs[0, 1].set_title('Mean')
                            axs[1, 0].imshow(std_ftmap, cmap=one_ch_cmap)
                            axs[1, 0].axis('off')
                            axs[1, 0].set_title('Std')
                            axs[1, 1].imshow(labels_as_image_spatial_info, cmap='tab20')
                            axs[1, 1].axis('off')
                            axs[1, 1].set_title('Cluster')
                            plt.tight_layout()
                            plt.savefig(prueba_ahora_path / f'subplots_{c:0>3}.png', dpi=300, bbox_inches='tight')
                            plt.close()

                            # Create clusters for different lambdas
                            lambda_imgs = []
                            for lmb in [0.8,0.5,0.2]:
                                samples_pairwise_distances_yx = pairwise_distances(flattened_ftmaps_with_yx[:, -2:], metric='euclidean')
                                samples_pairwise_distances_yx = (samples_pairwise_distances_yx - samples_pairwise_distances_yx.min()) / (samples_pairwise_distances_yx.max() - samples_pairwise_distances_yx.min())
                                samples_pairwise_distances_yx = samples_pairwise_distances_yx * (std_ft_distance + mean_ft_distance - (mean_ft_distance - std_ft_distance)) + (mean_ft_distance - std_ft_distance)
                                samples_pairwise_distances_weighted = lmb * samples_pairwise_distances_features + (1 - lmb) * samples_pairwise_distances_yx
                                db = DBSCAN(eps=eps, min_samples=16).fit(samples_pairwise_distances_weighted)
                                labels = db.labels_
                                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                                print(f"Number of clusters for lambda {lmb}: {n_clusters_}")
                                # Plot the labels as an image in the original size
                                labels_as_image = labels.reshape(ftmaps.shape[1], ftmaps.shape[2])
                                # Add padding to the labels
                                labels_as_image = np.pad(labels_as_image, ((y_padding, y_padding), (x_padding, x_padding)), 'constant', constant_values=(-1, -1))
                                lambda_imgs.append(labels_as_image)

                            # Create a PNG of good quality with 4 subplots, the original image, the mean, the std and the clustering
                            # with no axis and no white space
                            fig, axs = plt.subplots(2, 2, figsize=(8, 8))
                            axs[0, 0].imshow(orig_annotated.permute(1,2,0))
                            axs[0, 0].axis('off')
                            axs[0, 0].set_title('Original (annotated)')
                            axs[0, 1].imshow(lambda_imgs[2], cmap='tab20')
                            axs[0, 1].axis('off')
                            axs[0, 1].set_title('Lambda 0.2')
                            axs[1, 0].imshow(lambda_imgs[1], cmap='tab20')
                            axs[1, 0].axis('off')
                            axs[0, 1].set_title('Lambda 0.5')
                            axs[1, 1].imshow(lambda_imgs[0], cmap='tab20')
                            axs[1, 1].axis('off')
                            axs[0, 1].set_title('Lambda 0.8')
                            plt.tight_layout()
                            plt.savefig(prueba_ahora_path / f'subplots_{c:0>3}_lambdas.png', dpi=300, bbox_inches='tight')
                            plt.close()

                            # Skip to next iteration

                        if "pngs" in modos_de_visualizacion:
                            pass

                        if "bboxes" in modos_de_visualizacion:
                            # Max - Mean along the feature axis
                            max_of_evey_ftmap = ftmaps.max(axis=(1,2))
                            mean_of_evey_ftmap = ftmaps.mean(axis=(1,2))
                            ftmaps_minus_mean = ftmaps - mean_of_evey_ftmap[:, None, None]
                            # Sum of abs values
                            ftmaps_minus_mean_sum_abs = np.abs(ftmaps_minus_mean).sum(axis=0)
                            # Generate the image and bouding boxes
                            generate_image_with_bboxes(
                                saliency_map=ftmaps_minus_mean_sum_abs, folder_path=ftmaps_path,
                                original_image=imgs[_img_idx].permute(1,2,0).cpu().numpy(), padding=(y_padding, x_padding),
                                option='bboxes_with_oods',
                                ftmaps=ftmaps,
                                ood_method=self,
                            )
                            

                        if "ftmaps" in modos_de_visualizacion:
                            # Plot histogram of the feature maps mean and std
                            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                            axs[0].hist(ftmaps.mean(axis=(1,2)), bins=50)
                            axs[0].set_title('Mean')
                            axs[1].hist(ftmaps.std(axis=(1,2)), bins=50)
                            axs[1].set_title('Std')
                            plt.tight_layout()
                            plt.savefig(prueba_ahora_path / f'hist.png', dpi=300, bbox_inches='tight')
                            plt.close()

                            # Print the indices of topK feature maps by mean, std and max
                            topK = 50
                            # topK_indices = np.argsort(ftmaps.mean(axis=(1,2)))[::-1][:topK]
                            # print(f"Top {topK} feature maps by mean: {topK_indices}")
                            # topK_indices = np.argsort(ftmaps.std(axis=(1,2)))[::-1][:topK]
                            # print(f"Top {topK} feature maps by std: {topK_indices}")
                            # topK_indices = np.argsort(ftmaps.max(axis=(1,2)))[::-1][:topK]
                            # print(f"Top {topK} feature maps by max: {topK_indices}")
                            # topK_indices = np.argsort(weighted_variance(ftmaps))[::-1][:topK]
                            # print(f"Top {topK} feature maps by weighted variance: {topK_indices}")

                            # Spatial_freq
                            topK_indices = np.argsort(spatial_frequency_analysis(ftmaps))[:topK]  # Take the lowest spatial frequency
                            topK_ftmaps = ftmaps[topK_indices]
                            topK_ftmaps_sum = topK_ftmaps.sum(axis=0)
                            np.savetxt(ftmaps_path / F'txt_top{topK}_ftmaps_sum_spatial_freqs.txt', topK_ftmaps_sum)
                            topK_ftmaps_sum = (topK_ftmaps_sum - topK_ftmaps_sum.min()) / (topK_ftmaps_sum.max() - topK_ftmaps_sum.min())
                            topK_ftmaps_sum = np.pad(topK_ftmaps_sum, ((y_padding, y_padding), (x_padding, x_padding)), 'constant', constant_values=(0, 0))
                            plt.imshow(topK_ftmaps_sum, cmap=one_ch_cmap)
                            plt.savefig(ftmaps_path / f'top{topK}_ftmaps_sum_spatial_freqs.png', dpi=300, bbox_inches='tight')
                            plt.close()
                            

                            # Spread
                            topK_indices = np.argsort(center_of_mass_and_spread(ftmaps)[1])[:topK]  # Take the lowest spread
                            topK_ftmaps = ftmaps[topK_indices]
                            topK_ftmaps_sum = topK_ftmaps.sum(axis=0)
                            np.savetxt(ftmaps_path / F'txt_top{topK}_ftmaps_sum_spread.txt', topK_ftmaps_sum)
                            topK_ftmaps_sum = (topK_ftmaps_sum - topK_ftmaps_sum.min()) / (topK_ftmaps_sum.max() - topK_ftmaps_sum.min())
                            topK_ftmaps_sum = np.pad(topK_ftmaps_sum, ((y_padding, y_padding), (x_padding, x_padding)), 'constant', constant_values=(0, 0))
                            plt.imshow(topK_ftmaps_sum, cmap=one_ch_cmap)
                            plt.savefig(ftmaps_path / f'top{topK}_ftmaps_sum_spread.png', dpi=300, bbox_inches='tight')
                            plt.close()

                            # Entropy
                            topK_indices = np.argsort(entropy(ftmaps))[:topK]  # Take the lowest entropy
                            topK_ftmaps = ftmaps[topK_indices]
                            topK_ftmaps_sum = topK_ftmaps.sum(axis=0)
                            np.savetxt(ftmaps_path / F'txt_top{topK}_ftmaps_sum_entropy.txt', topK_ftmaps_sum)
                            topK_ftmaps_sum = (topK_ftmaps_sum - topK_ftmaps_sum.min()) / (topK_ftmaps_sum.max() - topK_ftmaps_sum.min())
                            topK_ftmaps_sum = np.pad(topK_ftmaps_sum, ((y_padding, y_padding), (x_padding, x_padding)), 'constant', constant_values=(0, 0))
                            plt.imshow(topK_ftmaps_sum, cmap=one_ch_cmap)
                            plt.savefig(ftmaps_path / f'top{topK}_ftmaps_sum_entropy.png', dpi=300, bbox_inches='tight')
                            plt.close()

                            # # Save the feature maps in another folder
                            # indiv_ftmaps_folder = ftmaps_path / 'individual_ftmaps'
                            # indiv_ftmaps_folder.mkdir(exist_ok=True)
                            # for idx_ftmap in range(ftmaps.shape[0]):

                            #     ftmap = ftmaps[idx_ftmap]

                            #     # Option 1: plot as it is
                            #     # plt.imshow(ftmap, vmin=-2, vmax=2, cmap='bwr')

                            #     # Option 2: resize and plot
                            #     # ftmap = resize(ftmap, (imgs[_img_idx].shape[1], imgs[_img_idx].shape[2]), anti_aliasing=True)
                            #     # plt.imshow(ftmap, vmin=-2, vmax=2, cmap='bwr')

                            #     # Option 2: scale and plot
                            #     ftmap = (ftmap - ftmap.min()) / (ftmap.max() - ftmap.min())
                            #     plt.imshow(ftmap, cmap=one_ch_cmap)

                            #     # Option 3: normalize to 0 - 1, resize and plot
                            #     # ftmap = (ftmap - ftmap.min()) / (ftmap.max() - ftmap.min())
                            #     # ftmap = resize(ftmap, (imgs[_img_idx].shape[1], imgs[_img_idx].shape[2]), anti_aliasing=True)
                            #     # plt.imshow(ftmap, cmap=one_ch_cmap)

                            #     # Save close
                            #     plt.savefig(indiv_ftmaps_folder / f'ftmap_{idx_ftmap}.pdf', bbox_inches='tight', dpi=300)
                            #     plt.close()
                            
                        
                        if "multiples_metricas" in modos_de_visualizacion:
                            # First save the original image
                            plt.imshow(imgs[_img_idx].cpu().permute(1,2,0))
                            plt.savefig(ftmaps_path / 'A_original.pdf')
                            plt.close()
                            # Save also the original with annotations
                            im = draw_bounding_boxes(
                                imgs[_img_idx].cpu(),
                                targets['bboxes'][_img_idx],
                                width=2,
                                font='FreeMonoBold',
                                font_size=11,
                                # Convert the classes to the names and above 19 all should be named unk
                                labels=[model.names[int(cls)] if cls <= 19 else "UNK" for cls in targets['cls'][_img_idx]],                        
                                colors=['blue']*len(targets['cls'][_img_idx])
                            )
                            plt.imshow(im.permute(1,2,0))
                            plt.savefig(ftmaps_path / 'A_original_with_annotations.pdf')
                            plt.close()
                            
                            # Make the mean of the feature maps
                            mean_ftmap = ftmaps.mean(axis=0)
                            mean_ftmap = (mean_ftmap - mean_ftmap.min()) / (mean_ftmap.max() - mean_ftmap.min())
                            # Add the padding
                            mean_ftmap = np.pad(mean_ftmap, ((y_padding, y_padding), (x_padding, x_padding)), 'constant', constant_values=(0, 0))
                            plt.imshow(mean_ftmap, cmap=one_ch_cmap)
                            plt.savefig(ftmaps_path / f'A_mean_ftmap.pdf')
                            plt.close()
                            mean_ftmap = resize(mean_ftmap, (imgs[_img_idx].shape[1], imgs[_img_idx].shape[2]), anti_aliasing=True)
                            plt.imshow(mean_ftmap, cmap=one_ch_cmap)
                            plt.savefig(ftmaps_path / f'A_mean_ftmap_reshaped.pdf')
                            plt.close()

                            # Make an image as the std of the feature maps
                            std_ftmap = ftmaps.std(axis=0)
                            std_ftmap = (std_ftmap - std_ftmap.min()) / (std_ftmap.max() - std_ftmap.min())
                            # Add the padding
                            std_ftmap = np.pad(std_ftmap, ((y_padding, y_padding), (x_padding, x_padding)), 'constant', constant_values=(0, 0))
                            plt.imshow(std_ftmap, cmap=one_ch_cmap)
                            plt.savefig(ftmaps_path / f'A_std_ftmap.pdf')
                            plt.close()
                            # Transform them to txt file and save it
                            np.savetxt(ftmaps_path / 'A_std_ftmap.txt', std_ftmap)
                            std_ftmap = resize(std_ftmap, (imgs[_img_idx].shape[1], imgs[_img_idx].shape[2]), anti_aliasing=True)
                            plt.imshow(std_ftmap, cmap=one_ch_cmap)
                            plt.savefig(ftmaps_path / f'A_std_ftmap_reshaped.pdf')
                            plt.close()

                            # Make an image as the max of the feature maps
                            max_ftmap = ftmaps.max(axis=0)
                            max_ftmap = (max_ftmap - max_ftmap.min()) / (max_ftmap.max() - max_ftmap.min())
                            # Add the padding
                            max_ftmap = np.pad(max_ftmap, ((y_padding, y_padding), (x_padding, x_padding)), 'constant', constant_values=(0, 0))
                            plt.imshow(max_ftmap, cmap=one_ch_cmap)
                            plt.savefig(ftmaps_path / f'A_max_ftmap.pdf')
                            plt.close()
                            # max_ftmap = resize(max_ftmap, (imgs[_img_idx].shape[1], imgs[_img_idx].shape[2]), anti_aliasing=True)
                            # plt.imshow(max_ftmap, cmap=one_ch_cmap)
                            # plt.savefig(ftmaps_path / f'A_max_ftmap_reshaped.pdf')
                            # plt.close()

                            # # Make an image as the min of the feature maps -> GIVES NO INFO
                            # min_ftmap = ftmaps.min(axis=0)
                            # min_ftmap = (min_ftmap - min_ftmap.min()) / (min_ftmap.max() - min_ftmap.min())
                            # # Add the padding
                            # min_ftmap = np.pad(min_ftmap, ((y_padding, y_padding), (x_padding, x_padding)), 'constant', constant_values=(0, 0))
                            # plt.imshow(min_ftmap, cmap=one_ch_cmap)
                            # plt.savefig(ftmaps_path / f'A_min_ftmap.pdf')
                            # plt.close()
                            # min_ftmap = resize(min_ftmap, (imgs[_img_idx].shape[1], imgs[_img_idx].shape[2]), anti_aliasing=True)
                            # plt.imshow(min_ftmap, cmap=one_ch_cmap)
                            # plt.savefig(ftmaps_path / f'A_min_ftmap_reshaped.pdf')
                            # plt.close()

                            # # Make an image of the IQR of the feature maps usign scipy.stats.iqr
                            # from scipy.stats import iqr
                            # iqr_ftmap = iqr(ftmaps, axis=0)
                            # iqr_ftmap = (iqr_ftmap - iqr_ftmap.min()) / (iqr_ftmap.max() - iqr_ftmap.min())
                            # # Add the padding
                            # iqr_ftmap = np.pad(iqr_ftmap, ((y_padding, y_padding), (x_padding, x_padding)), 'constant', constant_values=(0, 0))
                            # plt.imshow(iqr_ftmap, cmap=one_ch_cmap)
                            # plt.savefig(ftmaps_path / f'A_IQR_ftmap.pdf')
                            # plt.close()

                            # # Mean Absolute Deviation of the feature maps
                            # mean_ftmaps = ftmaps.mean(axis=0)
                            # mad_ftmap = np.mean(np.abs(ftmaps - mean_ftmaps), axis=0)
                            # mad_ftmap = (mad_ftmap - mad_ftmap.min()) / (mad_ftmap.max() - mad_ftmap.min())
                            # # Add the padding
                            # mad_ftmap = np.pad(mad_ftmap, ((y_padding, y_padding), (x_padding, x_padding)), 'constant', constant_values=(0, 0))
                            # plt.imshow(mad_ftmap, cmap=one_ch_cmap)
                            # plt.savefig(ftmaps_path / f'A_MAD_ftmap.pdf')
                            # plt.close()

                            # # Median Absolute Deviation (using scipy)
                            # mad_ftmap = sc_stats.median_abs_deviation(ftmaps, axis=0)
                            # mad_ftmap = (mad_ftmap - mad_ftmap.min()) / (mad_ftmap.max() - mad_ftmap.min())
                            # # Add the padding
                            # mad_ftmap = np.pad(mad_ftmap, ((y_padding, y_padding), (x_padding, x_padding)), 'constant', constant_values=(0, 0))
                            # plt.imshow(mad_ftmap, cmap=one_ch_cmap)
                            # plt.savefig(ftmaps_path / f'A_MedianAD_ftmap.pdf')
                            # plt.close()

                            # Max - mean of the feature maps
                            # Option 1: max along the pixel axis
                            max_minus_mean_ftmap = ftmaps.max(axis=0) - ftmaps.mean(axis=0)
                            np.savetxt(ftmaps_path / F'txt_A_max_minus_mean_ftmap.txt', max_minus_mean_ftmap)
                            max_minus_mean_ftmap = (max_minus_mean_ftmap - max_minus_mean_ftmap.min()) / (max_minus_mean_ftmap.max() - max_minus_mean_ftmap.min())
                            max_minus_mean_ftmap = np.pad(max_minus_mean_ftmap, ((y_padding, y_padding), (x_padding, x_padding)), 'constant', constant_values=(0, 0))
                            plt.imshow(max_minus_mean_ftmap, cmap=one_ch_cmap)
                            plt.savefig(ftmaps_path / f'A_max_minus_mean_ftmap.pdf')
                            plt.close()
                            # Option 2: max - mean along the feature axis
                            max_of_evey_ftmap = ftmaps.max(axis=(1,2))
                            mean_of_evey_ftmap = ftmaps.mean(axis=(1,2))
                            ftmaps_minus_mean = ftmaps - mean_of_evey_ftmap[:, None, None]
                            # 2.1 Sum of values
                            ftmaps_minus_mean_sum = ftmaps_minus_mean.sum(axis=0)
                            np.savetxt(ftmaps_path / F'txt_A_max_minus_mean_ftmap_sum.txt', ftmaps_minus_mean_sum)
                            ftmaps_minus_mean_sum = (ftmaps_minus_mean_sum - ftmaps_minus_mean_sum.min()) / (ftmaps_minus_mean_sum.max() - ftmaps_minus_mean_sum.min())
                            ftmaps_minus_mean_sum = np.pad(ftmaps_minus_mean_sum, ((y_padding, y_padding), (x_padding, x_padding)), 'constant', constant_values=(0, 0))
                            plt.imshow(ftmaps_minus_mean_sum, cmap=one_ch_cmap)
                            plt.savefig(ftmaps_path / f'A_max_minus_mean_ftmap_sum.pdf')
                            plt.close()
                            # 2.2 Sum of abs values
                            ftmaps_minus_mean_sum_abs = np.abs(ftmaps_minus_mean).sum(axis=0)
                            np.savetxt(ftmaps_path / F'txt_A_max_minus_mean_ftmap_sum_abs.txt', ftmaps_minus_mean_sum_abs)
                            ftmaps_minus_mean_sum_abs = (ftmaps_minus_mean_sum_abs - ftmaps_minus_mean_sum_abs.min()) / (ftmaps_minus_mean_sum_abs.max() - ftmaps_minus_mean_sum_abs.min())
                            ftmaps_minus_mean_sum_abs = np.pad(ftmaps_minus_mean_sum_abs, ((y_padding, y_padding), (x_padding, x_padding)), 'constant', constant_values=(0, 0))
                            plt.imshow(ftmaps_minus_mean_sum_abs, cmap=one_ch_cmap)
                            plt.savefig(ftmaps_path / f'A_max_minus_mean_ftmap_sum_abs.pdf')
                            plt.close()
                            
                            # Make an image of the sum of the topK values per pixel
                            #topK = 10
                            flattened_ftmaps = ftmaps.reshape(ftmaps.shape[0], -1).T
                            topK_folder_path = ftmaps_path / 'topK'
                            topK_folder_path.mkdir(exist_ok=True)
                            for topK in [1,3,5,7,10,15,20,25,30]:
                                topK_ftmap = np.sort(flattened_ftmaps, axis=1) # Sort the values (channels) of the pixels
                                topK_ftmap = topK_ftmap.reshape(ftmaps.shape[1], ftmaps.shape[2], -1)  # Reshape to the original shape
                                topK_ftmap = topK_ftmap[:,:, -topK:].sum(axis=2)  # Take the topK values of every pixel and sum them
                                topK_ftmap = (topK_ftmap - topK_ftmap.min()) / (topK_ftmap.max() - topK_ftmap.min())  # Normalize
                                # Add the padding
                                topK_ftmap = np.pad(topK_ftmap, ((y_padding, y_padding), (x_padding, x_padding)), 'constant', constant_values=(0, 0))
                                #topK_ftmap = topK_ftmap.reshape(ftmaps.shape[1], ftmaps.shape[2])
                                plt.imshow(topK_ftmap, cmap=one_ch_cmap)
                                plt.savefig(topK_folder_path / f'A_top{topK}_ftmap.pdf')
                                plt.close()
                                # topK_ftmap = resize(topK_ftmap, (imgs[_img_idx].shape[1], imgs[_img_idx].shape[2]), anti_aliasing=True)
                                # plt.imshow(topK_ftmap, cmap=one_ch_cmap)
                                # plt.savefig(topK_folder_path / f'A_top{topK}_ftmap_reshaped.pdf')
                                # plt.close()

                        if "clusters" in modos_de_visualizacion:
                            # CLuster
                            # OPTION 1. Directly clustering
                            features_metric = 'euclidean'
                            min_samples = 16
                            eps_options = {
                                "euclidean": 18,
                                "cosine": 0.1,
                                "manhattan": 18
                            }
                            eps = eps_options[features_metric]
                            # Flatten feature maps. original shape is (features, height, width). We want (n_samples, features)
                            flattened_ftmaps = ftmaps.reshape(ftmaps.shape[0], -1).T
                            db = DBSCAN(eps=eps, min_samples=min_samples, metric=features_metric).fit(flattened_ftmaps)
                            labels = db.labels_
                            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                            print(f"Number of clusters: {n_clusters_}")
                            # Plot the labels as an image in the original size
                            labels_as_image = labels.reshape(ftmaps.shape[1], ftmaps.shape[2])
                            plt.imshow(labels_as_image)
                            plt.savefig(ftmaps_path / f'A_cluster_dbscan.pdf')
                            plt.close()

                            # Scale the feature maps
                            from sklearn.preprocessing import StandardScaler
                            scaler = StandardScaler()
                            flattened_ftmaps_scaled = scaler.fit_transform(flattened_ftmaps)
                            # Cluster
                            db = DBSCAN(eps=eps, min_samples=min_samples).fit(flattened_ftmaps_scaled)
                            labels = db.labels_
                            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                            print(f"Number of clusters: {n_clusters_}")
                            # Plot the labels as an image in the original size
                            labels_as_image = labels.reshape(ftmaps.shape[1], ftmaps.shape[2])
                            plt.imshow(labels_as_image)
                            plt.savefig(ftmaps_path / f'A_cluster_dbscan_scaled.pdf')
                            plt.close()

                            # Clustering with DBSCAN but adding the spatial information
                            # Original feature map shape
                            num_channels, height, width = ftmaps.shape
                            # Create the coordinate channels
                            y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
                            # Add the coordinate channels to the original feature map
                            yx_augmented_feature_map = np.concatenate([ftmaps, y_coords[None, :, :], x_coords[None, :, :]])
                            # Now flatten the feature maps and make the first dimension be the pixels
                            flattened_ftmaps_with_yx = yx_augmented_feature_map.reshape(num_channels + 2, -1).T
                            samples_pairwise_distances_features = pairwise_distances(flattened_ftmaps_with_yx[:, :-2], metric=features_metric)
                            mean_ft_distance = samples_pairwise_distances_features.mean()
                            std_ft_distance = samples_pairwise_distances_features.std()
                            # Euclidean distance between sample coordinates. Normalize them to be in the same range as the feature distances
                            samples_pairwise_distances_yx = pairwise_distances(flattened_ftmaps_with_yx[:, -2:], metric='euclidean')
                            samples_pairwise_distances_yx = (samples_pairwise_distances_yx - samples_pairwise_distances_yx.min()) / (samples_pairwise_distances_yx.max() - samples_pairwise_distances_yx.min())
                            samples_pairwise_distances_yx = samples_pairwise_distances_yx * (std_ft_distance + mean_ft_distance - (mean_ft_distance - std_ft_distance)) + (mean_ft_distance - std_ft_distance)
                            # Merge the two distances using a weight lambda
                            lambda_weight = 0.5
                            samples_pairwise_distances_weighted = lambda_weight * samples_pairwise_distances_features + (1 - lambda_weight) * samples_pairwise_distances_yx
                            # Now cluster 
                            db = DBSCAN(eps=eps, min_samples=min_samples).fit(samples_pairwise_distances_weighted)
                            labels = db.labels_
                            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                            print(f"Number of clusters: {n_clusters_}")
                            # Plot the labels as an image in the original size
                            labels_as_image = labels.reshape(ftmaps.shape[1], ftmaps.shape[2])
                            # Add padding to the labels
                            labels_as_image = np.pad(labels_as_image, ((y_padding, y_padding), (x_padding, x_padding)), 'constant', constant_values=(-1, -1))
                            labels_as_image_spatial_info = labels_as_image.copy()
                            # As matrix
                            plt.matshow(labels_as_image, cmap='tab20')
                            # Plot also the legend
                            plt.savefig(ftmaps_path / f'A_cluster_with_spatial_info_matshow.pdf')
                            plt.close()
                            # As image
                            plt.imshow(labels_as_image, cmap='viridis')
                            plt.savefig(ftmaps_path / f'A_cluster_with_spatial_info_imshow.pdf')
                            plt.close()
                            # As image resized to the original size but maintaining the clusters
                            labels_as_image = resize(labels_as_image, (imgs[_img_idx].shape[1], imgs[_img_idx].shape[2]), anti_aliasing=True)
                            labels_as_image = np.rint(labels_as_image)                     
                            plt.imshow(labels_as_image, cmap='tab20')
                            plt.savefig(ftmaps_path / f'A_cluster_with_spatial_info_imshow_reshaped.pdf')
                            plt.close()

                        if "thresholds" in modos_de_visualizacion:
                            # Creating boxes with cv2 and std_ftmap
                            import cv2
                            # Remove one extra pixel from the padding
                            # ftmaps_one_less_pixel = ftmaps[:, 1:-1, 1:-1]
                            # std_ftmap = ftmaps_one_less_pixel.std(axis=0)
                            std_ftmap = ftmaps.std(axis=0)
                            std_ftmap = (std_ftmap - std_ftmap.min()) / (std_ftmap.max() - std_ftmap.min())
                            #std_ftmap = np.pad(std_ftmap, ((y_padding, y_padding), (x_padding, x_padding)), 'constant', constant_values=(0, 0))
                            # Threshold the map
                            saliency_map_8bit = np.uint8(std_ftmap * 255)
                            _, binary_map = cv2.threshold(saliency_map_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                            plt.imshow(binary_map, cmap=one_ch_cmap)
                            plt.savefig(ftmaps_path / f'A_binary_map_otsu.pdf')
                            plt.close()

                            # Adaptive Mean Thresholding
                            adaptive_mean = cv2.adaptiveThreshold(saliency_map_8bit, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
                            plt.imshow(adaptive_mean, cmap=one_ch_cmap)
                            plt.savefig(ftmaps_path / f'A_binary_map_adaptive_mean.pdf')
                            plt.close()

                            # Adaptive Gaussian Thresholding
                            adaptive_gaussian = cv2.adaptiveThreshold(saliency_map_8bit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                            plt.imshow(adaptive_gaussian, cmap=one_ch_cmap)
                            plt.savefig(ftmaps_path / f'A_binary_map_adaptive_gaussian.pdf')
                            plt.close()

                            # Triangle thresholding
                            _, triangle_threshold = cv2.threshold(saliency_map_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
                            plt.imshow(triangle_threshold, cmap=one_ch_cmap)
                            plt.savefig(ftmaps_path / f'A_binary_map_triangle.pdf')
                            plt.close()

                            from skimage.filters import threshold_multiotsu
                            # Applying Multi-Otsu threshold for the values in image
                            thresholds = threshold_multiotsu(saliency_map_8bit, classes=3)
                            multi_otsu_result = np.digitize(saliency_map_8bit, bins=thresholds)
                            plt.imshow(multi_otsu_result, cmap=one_ch_cmap)
                            plt.savefig(ftmaps_path / f'A_binary_map_multi_otsu.pdf')
                            plt.close()

                        # Selective search for bounding boxes (opencv)
                        # TODO: Deactivated for the moment
                        if False:
                            ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
                            ss.setBaseImage(saliency_map_8bit)
                            ss.switchToSelectiveSearchFast()
                            #ss.switchToSelectiveSearchQuality()
                            # run selective search on the input image
                            import time
                            start = time.time()
                            rects = ss.process()
                            end = time.time()
                            # show how along selective search took to run along with the total
                            # number of returned region proposals
                            print("[INFO] selective search took {:.4f} seconds".format(end - start))
                            print("[INFO] {} total region proposals".format(len(rects)))

                            # loop over the region proposals in chunks (so we can better visualize them)
                            import random
                            for i in range(0, len(rects), 100):
                                # clone the original image so we can draw on it
                                output = saliency_map_8bit.copy()
                                # loop over the current subset of region proposals
                                for (x, y, w, h) in rects[i:i + 100]:
                                    # draw the region proposal bounding box on the image
                                    color = [random.randint(0, 255) for j in range(0, 3)]
                                    cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
                            # Find contours
                            contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            # Draw bounding boxes
                            output_image = cv2.cvtColor(saliency_map_8bit, cv2.COLOR_GRAY2BGR)  # Convert to BGR for coloring
                            for contour in contours:
                                x, y, w, h = cv2.boundingRect(contour)
                                cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            
                            plt.imshow(output_image)
                            plt.savefig(ftmaps_path / f'A_boxes_from_std_ftmap.pdf')
                            plt.close()
                            # Save the result in an image with cv2 (same path as with matplotlib)
                            cv2.imwrite(str(ftmaps_path / f'A_boxes_from_std_ftmap_cv2.png'), output_image)

                        if False:
                            # OPTION 2. Clustering algorithms
                            # CLUSTERING OPTIONS:
                            cluster_option = 'optics'
                            if cluster_option == 'meanshift':
                                from sklearn.cluster import MeanShift
                                mean_shift = MeanShift(bandwidth=None)  # Experiment with the bandwidth parameter
                                labels = mean_shift.fit_predict(flattened_ftmaps_with_xy_scaled)
                            elif cluster_option == 'dbscan':
                                db = DBSCAN(eps=0.9, min_samples=8).fit(flattened_ftmaps_with_xy_scaled)
                                labels = db.labels_
                            elif cluster_option == 'gmm':
                                from sklearn.mixture import GaussianMixture
                                gmm = GaussianMixture(n_components=5, covariance_type='full')
                                labels = gmm.fit_predict(flattened_ftmaps_with_xy_scaled)
                            elif cluster_option == 'optics':
                                from sklearn.cluster import OPTICS
                                optics = OPTICS(min_samples=8, xi=.05, min_cluster_size=.05)
                                optics_labels = optics.fit_predict(flattened_ftmaps_with_xy_scaled)
                                print(f"OPTICS found {len(np.unique(optics_labels))} clusters")
                            else:
                                raise ValueError("The cluster option is not valid")
                            # Number of clusters
                            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                            print(f"Number of clusters: {n_clusters_}")
                            # Plot the labels as an image in the original size
                            labels_as_image = labels.reshape(ftmaps_shape[1], ftmaps_shape[2])
                            plt.imshow(labels_as_image, cmap='tab20')
                            # Plot also the legend
                            plt.colorbar()
                            plt.savefig(ftmaps_path / f'A_cluster_with_spatial_info_{cluster_option}.pdf')
                            plt.close()                            

                print()
                if c > 20:
                    quit()

            elif debugeando_en_modo == "normal":
                # Por aqui va la ejecucion normal

                ### Comprobar si las cajas predichas son OoD ###
                ood_decision = self.compute_ood_decision_on_results(results, logger)

                ### Añadir posibles cajas desconocidas a las predicciones ###
                if self.enhanced_unk_localization:
                    possible_unk_bboxes, ood_decision_on_unknown = self.compute_extra_possible_unkwnown_bboxes(results, logger)
            
                plot_results( 
                    class_names=model.names,
                    results=results,
                    folder_path=folder_path,
                    now=now,
                    valid_preds_only=False,
                    origin_of_idx=idx_of_batch*dataloader.batch_size,
                    image_format='pdf',
                    ood_decision=ood_decision,
                    ood_method_name=self.name,
                    targets=targets
                )
            else:
                raise ValueError("The mode to debug is not valid")
            
            number_of_images_saved += len(data['im_file'])
            # TODO: De momento no queremos plotear todo, solo unos pocos batches
            if number_of_images_saved > 200:
                quit()
            # if idx_of_batch > 10:
            #     quit()
                
    def iterate_data_to_compute_metrics(self, model: YOLO, device: str, dataloader: InfiniteDataLoader, logger, known_classes: List[int]) -> Dict[str, float]:

        logger.warning(f"Using a confidence threshold of {self.min_conf_threshold} for tests")
        number_of_images_processed = 0
        number_of_batches = len(dataloader)
        all_preds = []
        all_targets = []
        assert hasattr(dataloader.dataset, "number_of_classes"), "The dataset does not have the attribute number_of_classes to know the number of classes known in the dataset"
        class_names = list(dataloader.dataset.data['names'].values())[:dataloader.dataset.number_of_classes]
        class_names.append('unknown')
        known_classes_tensor = Tensor(known_classes, dtype=torch.float32)
        for idx_of_batch, data in enumerate(dataloader):

            if idx_of_batch % 50 == 0 or idx_of_batch == number_of_batches - 1:
                logger.info(f"{(idx_of_batch/number_of_batches) * 100:02.1f}%: Procesing batch {idx_of_batch+1} of {number_of_batches}") 

            ### Preparar imagenes y targets ###
            imgs, targets = self.prepare_data_for_model(data, device)
            
            ### Procesar imagenes en el modelo para obtener logits y las cajas ###
            results = model.predict(imgs, save=False, verbose=False, conf=self.min_conf_threshold)

            ### Comprobar si las cajas predichas son OoD ###
            ood_decision = self.compute_ood_decision_on_results(results, logger)

            ### Añadir posibles cajas desconocidas a las predicciones ###
            if self.enhanced_unk_localization:
                possible_unk_bboxes, ood_decision_on_unknown = self.compute_extra_possible_unkwnown_bboxes(results, logger)

            # Cada prediccion va a ser un diccionario con las siguientes claves:
            #   'img_idx': int -> Indice de la imagen
            #   'img_name': str -> Nombre del archivo de la imagen
            #   'bboxes': List[Tensor] -> Lista de tensores con las cajas predichas
            #   'cls': List[Tensor] -> Lista de tensores con las clases predichas
            #   'conf': List[Tensor] -> Lista de tensores con las confianzas de las predicciones (en yolov8 es cls)
            #   'ood_decision': List[int] -> Lista de enteros con la decision de si la caja es OoD o no
            for img_idx, res in enumerate(results):
                #for idx_bbox in range(len(res.boxes.cls)):
                # Parse the ood elements as the unknown class (80)
                ood_decision_one_image = Tensor(ood_decision[img_idx], dtype=torch.float32)
                unknown_mask = ood_decision_one_image == 0
                bboxes_cls = torch.where(unknown_mask, Tensor(80, dtype=torch.float32), res.boxes.cls.cpu())
                all_preds.append({
                    'img_idx': number_of_images_processed + img_idx,
                    'img_name': Path(data['im_file'][img_idx]).stem,
                    'bboxes': res.boxes.xyxy,
                    'cls': bboxes_cls,
                    'conf': res.boxes.conf,
                    'ood_decision': Tensor(ood_decision[img_idx], dtype=torch.float32)
                })
                # Transform the classes to index 80 if they are not in the known classes
                known_mask = torch.isin(targets['cls'][img_idx], known_classes_tensor)
                transformed_target_cls = torch.where(known_mask, targets['cls'][img_idx], Tensor(80, dtype=torch.float32))
                all_targets.append({
                    'img_idx': number_of_images_processed + img_idx,
                    'img_name': Path(data['im_file'][img_idx]).stem,
                    'bboxes': targets['bboxes'][img_idx],
                    'cls': transformed_target_cls
                })

            ### Acumular predicciones y targets ###
            number_of_images_processed += len(imgs)

            # # Plot one image of the batch with predictions and targets
            # idx_image = 4
            # plot_results(
            #     class_names=model.names,
            #     results=results,
            #     folder_path=Path('.'),
            #     now='ahora',
            #     valid_preds_only=False,
            #     origin_of_idx=idx_of_batch*dataloader.batch_size,
            #     image_format='pdf',
            #     #ood_decision=ood_decision,
            #     targets=targets,
            # )

        # All predictions collected, now compute metrics
        results_dict = compute_metrics(all_preds, all_targets, class_names, known_classes, logger)

        # Count the number of non-unknown instances and the number of unknown instances
        number_of_known_boxes = 0 
        number_of_unknown_boxes = 0
        for _target in all_targets:
            number_of_known_boxes += torch.sum(_target['cls'] != 80).item()
            number_of_unknown_boxes += torch.sum(_target['cls'] == 80).item()
        logger.info(f"Number of known boxes: {number_of_known_boxes}")
        logger.info(f"Number of unknown boxes: {number_of_unknown_boxes}")

        return results_dict

    
    def generate_thresholds(self, ind_scores: list, tpr: float, logger) -> Union[List[float], List[List[float]]]:
        """
        Generate the thresholds for each class using the in-distribution scores.
        If per_class=True, in_scores must be a list of lists,
        where each list is the list of scores for each class.
        tpr must be in the range [0, 1]
        """
        if self.distance_method:
            # If the method measures distance, the higher the score, the more OOD. Therefore
            # we need to get the upper bound, the tpr*100%
            used_tpr = 100*tpr
        else:            
            # As the method is a similarity method, the higher the score, the more IND. Therefore
            # we need to get the lower bound, the (1-tpr)*100%
            used_tpr = (1 - tpr)*100

        sufficient_samples = 10
        good_number_of_samples = 50

        if self.per_class:

            if self.per_stride:

                # Per class with stride differentiation
                thresholds = [[[] for _ in range(3)] for _ in range(len(ind_scores))]
                for idx_cls, ind_scores_one_cls in enumerate(ind_scores):
                    for idx_stride, ind_scores_one_cls_one_stride in enumerate(ind_scores_one_cls):
                        if len(ind_scores_one_cls_one_stride) > sufficient_samples:
                            thresholds[idx_cls][idx_stride] = float(np.percentile(ind_scores_one_cls_one_stride, used_tpr, method='lower'))
                            if len(ind_scores_one_cls_one_stride) < good_number_of_samples:
                                logger.warning(f"Class {idx_cls:03}, Stride {idx_stride}: has {len(ind_scores_one_cls_one_stride)} samples. The threshold may not be accurate")
                        else:
                            logger.warning(f'Class {idx_cls:03}, Stride {idx_stride} -> Has less than {sufficient_samples} samples. No threshold is generated')
            
            else:

                # Per class with no stride differentiation
                thresholds = [0 for _ in range(len(ind_scores))]
                for idx, cl_scores in enumerate(ind_scores):
                    if len(cl_scores) > sufficient_samples:
                        thresholds[idx] = float(np.percentile(cl_scores, used_tpr, method='lower'))
                        if len(cl_scores) < good_number_of_samples:
                            logger.warning(f"Class {idx}: {len(cl_scores)} samples. The threshold may not be accurate")
                    else:
                        logger.warning(f"Class {idx} has less than {sufficient_samples} samples. No threshold is generated")
                        
        else:
            raise NotImplementedError("Not implemented yet")
        
        return thresholds
    
    def compute_extra_possible_unkwnown_bboxes(self, feature_maps: Union[Results, Tensor, np.ndarray], logger: Logger) -> Tuple[List[Tensor], List[int]]:
        """
        Compute the possible unknown bounding boxes using the feature maps of the model.
        """
        if isinstance(feature_maps, Results):
            pass
        elif isinstance(feature_maps, Tensor):
            pass
        elif isinstance(feature_maps, np.ndarray):
            pass

        

#################################################################################
# Create classes for each method. Methods will inherit from OODMethod,
#   will override the abstract methods and also any other function that is needed.
#################################################################################

### Superclasses for methods using logits of the model ###
class LogitsMethod(OODMethod):
    
    def __init__(self, name: str, per_class: bool, per_stride: bool, iou_threshold_for_matching: float, min_conf_threshold: float, **kwargs):
        distance_method = False
        which_internal_activations = 'logits'  # Always logits for these methods
        enhanced_unk_localization = False  # By default not used with logits, as feature maps are needed.
        super().__init__(name, distance_method, per_class, per_stride, iou_threshold_for_matching, min_conf_threshold, which_internal_activations, enhanced_unk_localization)

    def compute_ood_decision_on_results(self, results: Results, logger) -> List[List[int]]:
        ood_decision = []  
        for idx_img, res in enumerate(results):
            ood_decision.append([])  # Every image has a list of decisions for each bbox
            for idx_bbox in range(len(res.boxes.cls)):
                cls_idx = int(res.boxes.cls[idx_bbox].cpu())
                logits = res.extra_item[idx_bbox][4:].cpu()
                score = self.compute_score_one_bbox(logits, cls_idx)
                if score < self.thresholds[cls_idx]:
                    ood_decision[idx_img].append(0)  # OOD
                else:
                    ood_decision[idx_img].append(1)  # InD

        return ood_decision
    
    def extract_internal_activations(self, results: Results, all_activations: List[float], targets: Dict[str, Tensor]):
        """
        The extracted activations will be stored in the list all_activations. 
        In this case, the scores are directly computed.
        """
        for res in results:
            # Loop over the valid predictions
            for valid_idx_one_bbox in res.valid_preds:
                cls_idx_one_bbox = int(res.boxes.cls[valid_idx_one_bbox].cpu())
                logits_one_bbox = res.extra_item[valid_idx_one_bbox][4:].cpu()
                all_activations[cls_idx_one_bbox].append(self.compute_score_one_bbox(logits_one_bbox, cls_idx_one_bbox))

    def format_internal_activations(self, all_activations: Union[List[float], List[List[np.ndarray]]]):
        """
        Format the internal activations of the model. In this case, the activations are already well formatted.
        """
        pass


class MSP(LogitsMethod):

    def __init__(self, **kwargs):
        name = 'MSP'
        super().__init__(name, **kwargs)
    
    def compute_score_one_bbox(self, logits: Tensor, cls_idx: int) -> float:
        logits = logits.numpy()
        assert cls_idx == logits.argmax(), "The max logit is not the one of the predicted class"
        return logits[cls_idx]

    # def extract_internal_activations(self, results: Results, all_activations: List[float]):
    #     """
    #     The extracted activations will be stored in the list all_activations
    #     """
    #     for res in results:
    #         # Loop over the valid predictions
    #         for valid_idx_one_bbox in res.valid_preds:
                
    #             cls_idx_one_bbox = int(res.boxes.cls[valid_idx_one_bbox].cpu())

    #             logits_one_bbox = res.extra_item[valid_idx_one_bbox][4:].cpu().numpy()

    #             all_activations[cls_idx_one_bbox].append(logits_one_bbox.max())

    
class Energy(LogitsMethod):

    temper: float

    def __init__(self, temper: float, **kwargs):
        name = 'Energy'
        super().__init__(name, **kwargs)
        self.temper = temper
    
    def compute_score_one_bbox(self, logits, cls_idx) -> float:
        return self.temper * torch.logsumexp(logits / self.temper, dim=0).item()

    # def extract_internal_activations(self, results: Results, all_activations: List[float]):
    #     """
    #     The extracted activations will be stored in the list all_activations
    #     """
    #     for res in results:
    #         # Loop over the valid predictions
    #         for valid_idx_one_bbox in res.valid_preds:
                
    #             cls_idx_one_bbox = int(res.boxes.cls[valid_idx_one_bbox].cpu())

    #             logits_one_bbox = res.extra_item[valid_idx_one_bbox][4:].cpu()

    #             all_activations[cls_idx_one_bbox].append(self.temper * torch.logsumexp(logits_one_bbox / self.temper, dim=0).item())


### Superclasses for methods using feature maps of the model ###
class DistanceMethod(OODMethod):
    
    agg_method: Callable
    cluster_method: str
    cluster_optimization_metric: str
    available_cluster_methods: List[str]
    available_cluster_optimization_metrics: List[str]
    clusters: Union[List[np.ndarray], List[List[np.ndarray]]]
    ind_info_creation_option: str
    enhanced_unk_localization: bool

    # name: str, distance_method: bool, per_class: bool, per_stride: bool, iou_threshold_for_matching: float, min_conf_threshold: float
    # def __init__(self, name: str, agg_method: str, per_class: bool, per_stride: bool, cluster_method: str,
    #              cluster_optimization_metric: str, ind_info_creation_option: str, **kwargs):
    def __init__(self, name: str, per_class: bool, per_stride: bool, cluster_method: str,
                 cluster_optimization_metric: str, agg_method: str, ind_info_creation_option: str, which_internal_activations: str, **kwargs):
        distance_method = True  # Always True for distance methods
        which_internal_activations = self.validate_correct_which_internal_activations_distance_methods(which_internal_activations)
        super().__init__(name, distance_method, per_class, per_stride, which_internal_activations=which_internal_activations, **kwargs)
        self.cluster_method = self.check_cluster_method_selected(cluster_method)
        self.cluster_optimization_metric = self.check_cluster_optimization_metric_selected(cluster_optimization_metric)
        self.agg_method = self.select_agg_method(agg_method)
        self.ind_info_creation_option = self.validate_correct_ind_info_creation_option(ind_info_creation_option)
        # if agg_method == 'mean':
        #     self.agg_method = np.mean
        # elif agg_method == 'median':
        #     self.agg_method = np.median
        # else:
        #     raise NameError(f"The agg_method argument must be one of the following: 'mean', 'median'. Current value: {agg_method}")
    
    def validate_correct_which_internal_activations_distance_methods(self, which_internal_activations: str) -> str:
        assert which_internal_activations in FTMAPS_RELATED_OPTIONS, f"which_internal_activations must be one of {FTMAPS_RELATED_OPTIONS}, but got {which_internal_activations}"
        return which_internal_activations
        
    def validate_correct_ind_info_creation_option(self, ind_info_creation_option: str) -> str:
        assert ind_info_creation_option in IND_INFO_CREATION_OPTIONS, f"ind_info_creation_option must be one of {IND_INFO_CREATION_OPTIONS}, but got {ind_info_creation_option}"
        return ind_info_creation_option

    def select_agg_method(self, agg_method: str) -> Callable:
        assert agg_method in ['mean', 'median'], f"agg_method must be one of ['mean', 'median'], but got {agg_method}"
        return np.mean if agg_method == 'mean' else np.median
        
    # TODO: Quiza estas formulas acaben devolviendo un Callable con el propio método que implementen
    def check_cluster_method_selected(self, cluster_method: str) -> str:
        assert cluster_method in AVAILABLE_CLUSTERING_METHODS, f"cluster_method must be one of {AVAILABLE_CLUSTERING_METHODS}, but got {cluster_method}"
        return cluster_method

    def check_cluster_optimization_metric_selected(self, cluster_optimization_metric: str) -> str:
        assert cluster_optimization_metric in AVAILABLE_CLUSTER_OPTIMIZATION_METRICS, f"cluster_method must be one" \
          f"of {AVAILABLE_CLUSTER_OPTIMIZATION_METRICS}, but got {cluster_optimization_metric}"
        return cluster_optimization_metric

    def compute_score_one_bbox(self, cluster: np.array, activations: np.array) -> List[float]:
        return self.compute_distance(cluster, activations)

    # TODO: Juntar las dos funciones de debajo en una sola, ya que podemos devolver una lista de floats siempre
    #   y en el caso de que solo hayamos enviado una bbox, la lista tendra un solo elemento
    @abstractmethod
    def compute_distance(self, cluster: np.array, activations: np.array) -> List[float]:
        """
        """
        pass

    def extract_internal_activations(self, results: Results, all_activations: List[List[List[np.ndarray]]], targets: Dict[str, Tensor]):
        """
        Extract the ftmaps of the selected boxes in their corresponding stride and class.
        This function must have all the logic to extract the internal activations of the model depeding on 
        which internal activations are selected and which in-distribution info creation option is selected.
        The extracted activations will be stored in the list all_activations with the following structure:
            all_activations = [
                [  # Class 0
                    [  # Stride 0
                        ndarray[C, H, W],  # Bbox 0 of class 0 in stride 0
                        ndarray[C, H, W],  # Bbox 1 of class 0 in stride 0
                        ...
                    ],
                    [  # Stride 1
                        ndarray[C, H, W],  # Bbox 0 of class 0 in stride 1
                        ndarray[C, H, W],  # Bbox 1 of class 0 in stride 1
                        ...
                    ],
                    ...
                ],
                [ # Class 1
                    ...
                ],
                ...
            ]
        """
        device = results[0].extra_item[0][0].device

        ### Case where we have extracted the full feature maps ###
        if self.which_internal_activations == 'ftmaps_and_strides':
            if self.ind_info_creation_option in ['all_targets_one_stride', 'all_targets_all_strides']:
                for idx_img in range(len(targets["bboxes"])):
                    # Use target bboxes to extract the ftmaps
                    cls_idx_one_img = targets["cls"][idx_img]
                    boxes_one_img = targets["bboxes"][idx_img].to(device).float()
                    ftmaps, strides = results[idx_img].extra_item
                    if self.ind_info_creation_option == 'all_targets_all_strides':
                        roi_aligned_ftmaps_per_stride = extract_roi_aligned_features_from_correct_stride(
                            ftmaps=[ft[None, ...] for ft in ftmaps],
                            boxes=[boxes_one_img],
                            strides=[strides],
                            img_shape=results[idx_img].orig_img.shape[2:],
                            device=device,
                            extract_all_strides=True,
                        )
                    else:
                        # TODO: Not really an interesting option, as if we do not have stride information we must infer it.
                        #   Anyway, we could be implement it
                        raise NotImplementedError("Not implemented yet")

                    # Add all the roi aligned ftmaps to the list
                    for stride_idx, (bbox_idx_in_one_stride, ftmaps_one_stride) in enumerate(roi_aligned_ftmaps_per_stride[0]):  # Batch size is 1
                        if len(bbox_idx_in_one_stride) > 0:
                            for i, bbox_idx in enumerate(bbox_idx_in_one_stride):
                                bbox_idx = bbox_idx.item()
                                target_cls = int(cls_idx_one_img[bbox_idx].item())
                                all_activations[target_cls][stride_idx].append(ftmaps_one_stride[i].cpu().numpy())

            elif self.ind_info_creation_option in ['valid_preds_one_stride', 'valid_preds_all_strides']:
                # Loop each image fo the batch
                for res in results:
                    cls_idx_one_pred = res.boxes.cls.cpu()
                    # Loop each stride and get only the ftmaps of the boxes that are valid predictions
                    ftmaps, strides = res.extra_item
                    valid_preds = res.valid_preds
                    # Use RoIAlign to extract the features of the predicted boxes
                    # As the function is created for processing a batch all in once, we need to 
                    # adapt the call by passing a batch of only 1 image
                    roi_aligned_ftmaps_per_stride = extract_roi_aligned_features_from_correct_stride(
                        ftmaps=[ft[None, ...] for ft in ftmaps],
                        boxes=[res.boxes.xyxy],  
                        strides=[strides],
                        img_shape=res.orig_img.shape[2:],
                        device=device
                    )[0]  # As we only introduce one image, we need to get the only element of the batch
                    
                    # Add the valid roi aligned ftmaps to the list
                    self._extract_valid_preds_from_one_image_roi_aligned_ftmaps(cls_idx_one_pred, roi_aligned_ftmaps_per_stride, valid_preds, all_activations)

            else:
                raise ValueError("Wrong ind_info_creation_option for the selected internal activations.")

        ### Case where only the RoIAligned feature maps are extracted ###
        elif self.which_internal_activations == 'roi_aligned_ftmaps':
            if self.ind_info_creation_option in ['all_targets_one_stride', 'all_targets_all_strides']:
                raise AssertionError("This options are only compatible if all the feature maps are extracted, not only the RoIAligned ones")

            # Loop each image fo the batch
            for res in results:
                cls_idx_one_pred = res.boxes.cls.cpu()
                roi_aligned_ftmaps_per_stride = res.extra_item
                valid_preds = res.valid_preds
                self._extract_valid_preds_from_one_image_roi_aligned_ftmaps(cls_idx_one_pred, roi_aligned_ftmaps_per_stride, valid_preds, all_activations)
                
                # # Loop each stride and get only the ftmaps of the boxes that are valid predictions
                # for stride_idx, (bbox_idx_in_one_stride, ftmaps) in enumerate(res.extra_item):
                #     if len(bbox_idx_in_one_stride) > 0:  # Check if there are any predictions in this stride
                #         for i, bbox_idx in enumerate(bbox_idx_in_one_stride):
                #             bbox_idx = bbox_idx.item() 
                #             if self.ind_info_creation_option == 'valid_preds_one_stride':
                #                 # Use only predictions that are "valid", i.e., correctly predicted and asociated univocally GT
                #                 # and use only the stride where the bbox is predicted
                #                 if bbox_idx in res.valid_preds:
                #                     pred_cls = int(cls_idx_one_pred[bbox_idx].item())
                #                     all_activations[pred_cls][stride_idx].append(ftmaps[i].cpu().numpy())
                #             elif self.ind_info_creation_option == 'valid_preds_all_strides':
                #                 # Use only predictions that are "valid", i.e., correctly predicted and asociated univocally GT
                #                 # and use all the strides 
                #                 raise NotImplementedError("Not implemented yet")
                            
        else:
            raise NotImplementedError("The method to extract internal activations is not implemented yet")

    def _extract_valid_preds_from_one_image_roi_aligned_ftmaps(
            self, cls_idx_one_pred: Tensor, roi_aligned_ftamps_per_stride: List[List[Tensor]], valid_predictions: List[int], all_activations: List[List[List[np.ndarray]]]
        ):
        for stride_idx, (bbox_idx_in_one_stride, ftmaps_one_stride) in enumerate(roi_aligned_ftamps_per_stride):
            if len(bbox_idx_in_one_stride) > 0:  # Check if there are any predictions in this stride
                for i, bbox_idx in enumerate(bbox_idx_in_one_stride):
                    bbox_idx = bbox_idx.item() 
                    if self.ind_info_creation_option == 'valid_preds_one_stride':
                        # Use only predictions that are "valid", i.e., correctly predicted and asociated univocally GT
                        # and use only the stride where the bbox is predicted
                        if bbox_idx in valid_predictions:
                            pred_cls = int(cls_idx_one_pred[bbox_idx].item())
                            all_activations[pred_cls][stride_idx].append(ftmaps_one_stride[i].cpu().numpy())
                    elif self.ind_info_creation_option == 'valid_preds_all_strides':
                        # Use only predictions that are "valid", i.e., correctly predicted and asociated univocally GT
                        # and use all the strides 
                        raise NotImplementedError("Not implemented yet")

    def format_internal_activations(self, all_activations: List[List[List[np.ndarray]]]):
        """
        Format the internal activations of the model. In this case, the ftmaps are converted to numpy arrays.
        The extracted activations will be stored in the list all_activations with the following structure:
            all_activations = [
                [  # Class 0
                    [  # Stride 0
                        ndarray[N_0_0, C, H, W],  # Bboxes of class 0 in stride 0
                        ...
                    ],
                    [  # Stride 1
                        ndarray[N_0_1, C, H, W],  # Bboxes of class 0 in stride 1
                        ...
                    ],
                    ...
                ],
                [ # Class 1
                    [  # Stride 0
                        ndarray[N_1_0, C, H, W],  # Bboxes of class 1 in stride 0
                        ...
                    ],
                    [  # Stride 1
                        ndarray[N_1_1, C, H, W],  # Bboxes of class 1 in stride 1
                        ...
                    ],
                ],
                ...
            ]
        N_0_0 is the number of bboxes of class 0 in stride 0, and so on.
        """
        # Convert the list inside each class and stride to numpy arrays
        for idx_cls, ftmaps_one_cls in enumerate(all_activations):
            for idx_stride, ftmaps_one_cls_one_stride in enumerate(ftmaps_one_cls):
                if len(ftmaps_one_cls_one_stride) > 0:
                    all_activations[idx_cls][idx_stride] = np.stack(ftmaps_one_cls_one_stride, axis=0)
                else:
                    all_activations[idx_cls][idx_stride] = np.empty(0)
   

    def compute_scores_from_activations(self, activations: Union[List[np.ndarray], List[List[np.ndarray]]], logger: Logger):
        """
        Compute the scores for each class using the in-distribution activations (usually feature maps). They come in form of a list of ndarrays when
            per_class True and per_stride are False, where each position of the list refers to one class and the array is a tensor of shape [N, C, H, W]. 
            When is per_class and per_stride, the first list refers to classes and the second to the strides, being the arrays of the same shape as presented.
        """
        if self.per_class:

            if self.per_stride:
                    
                scores = [[[] for _ in range(3)] for _ in range(len(activations))]

                if self.cluster_method == 'one':
                    self.compute_scores_one_cluster_per_class_and_stride(activations, scores, logger)
                else:
                    raise NotImplementedError("Not implemented yet")
                
            else:
                raise NotImplementedError("Not implemented yet")
            
        else:
            raise NotImplementedError("Not implemented yet")
        
        return scores
    
    # TODO: Esta funcion seguramente se pueda generalizar
    def compute_scores_one_cluster_per_class_and_stride(self, activations: List[List[np.ndarray]], scores: List[List[np.ndarray]], logger):
        """
        This function has the logic of looping over the classes and strides to then call the function that computes the scores on one class and one stride.
        """
        for idx_cls, activations_one_cls in enumerate(activations):

            logger.info(f'Class {idx_cls:03} of {len(activations)}')
            for idx_stride, activations_one_cls_one_stride in enumerate(activations_one_cls):
                
                if len(activations_one_cls_one_stride) > 0:

                    if len(self.clusters[idx_cls][idx_stride]) > 0:

                        scores[idx_cls][idx_stride] = self.compute_scores_one_class_one_stride(
                            self.clusters[idx_cls][idx_stride][None, :], 
                            self.activations_transformation(activations_one_cls_one_stride)
                            # activations_one_cls_one_stride.reshape(activations_one_cls_one_stride.shape[0], -1)
                        )

                    if len(activations_one_cls_one_stride) < 50:
                        logger.warning(f'WARNING: Class {idx_cls:03}, Stride {idx_stride} -> Only {len(activations_one_cls_one_stride)} samples')

                else:
                    logger.warning(f'SKIPPING Class {idx_cls:03}, Stride {idx_stride} -> NO SAMPLES')
                    scores[idx_cls][idx_stride] = np.empty(0)        

    def compute_scores_one_class_one_stride(self,clusters_one_cls_one_stride: np.array,  ind_activations_one_cls_one_stride: np.array) -> List[float]:
        """
        Compute the scores for one class using the in-distribution activations (usually feature maps).
        """
        scores = []
        if len(ind_activations_one_cls_one_stride) > 0:
            if len(clusters_one_cls_one_stride) > 0:
                scores = self.compute_distance(clusters_one_cls_one_stride, ind_activations_one_cls_one_stride)
            else:
                raise ValueError("The clusters must have at least one sample")
        return scores

    def compute_ood_decision_on_results(self, results: Results, logger) -> List[List[int]]:
        """
        Compute the OOD decision for each class using the in-distribution activations (usually feature maps).
        Pipeline:
            1. Loop over the results (predictions). Each result is from one image
            2. Compute the distance between the prediction and the cluster of the predicted class
            3. Compare the distance with the threshold
        """
        ood_decision = []
        if self.which_internal_activations == "ftmaps_and_strides":
            for idx_img, res in enumerate(results):
                ood_decision.append([])
                ftmaps, strides = res.extra_item
                roi_aligned_ftmaps_per_stride = extract_roi_aligned_features_from_correct_stride(
                            ftmaps=[ft[None, ...] for ft in one_stride_ftmaps],
                            boxes=[res.boxes.xyxy[stride_mask]],
                            strides=[strides[stride_mask]],
                            img_shape=res.orig_img.shape[2:],
                            device=res.boxes.xyxy.device,
                            extract_all_strides=False,
                        )
                

                for stride_idx, one_stride_ftmaps in enumerate(ftmaps):
                    stride_mask = strides == stride_idx
                    if stride_mask.any():  # Only enter if there are any predictions in this stride
                        print(f'stride_idx: {stride_idx}')
                        # Now RoIAlign the ftmaps of the stride
                       
                    

        elif self.which_internal_activations == 'roi_aligned_ftmaps':
            for idx_img, res in enumerate(results):
                ood_decision.append([])  # Every image has a decisions for each bbox
                for stride_idx, (bbox_idx_in_one_stride, ftmaps) in enumerate(res.extra_item):

                    if len(bbox_idx_in_one_stride) > 0:  # Only enter if there are any predictions in this stride
                        # Each ftmap is from a bbox prediction
                        for idx, ftmap in enumerate(ftmaps):
                            bbox_idx = idx
                            cls_idx = int(res.boxes.cls[bbox_idx].cpu())
                            ftmap = ftmap.cpu().unsqueeze(0).numpy()  # To obtain a tensor of shape [1, C, H, W]
                            # ftmap = ftmap.cpu().flatten().unsqueeze(0).numpy()
                            # [None, :] is to do the same as unsqueeze(0) but with numpy
                            if len(self.clusters[cls_idx][stride_idx]) == 0:
                                logger.warning(f'Image {idx_img}, bbox {bbox_idx} is viewed as an OOD.' \
                                                'It cannot be compared as there is no cluster for class {cls_idx} and stride {stride_idx')
                                distance = 1000
                            else:
                                distance = self.compute_distance(
                                    self.clusters[cls_idx][stride_idx][None, :], 
                                    self.activations_transformation(ftmap)
                                )[0]
                            
                            # d = pairwise_distances(clusters[cls_idx][stride_idx][None,:], ftmap.cpu().numpy().reshape(1, -1), metric='l1')

                            # print('------------------------------')
                            # print('idx_img:\t', idx_of_batch*dataloader.batch_size + idx_img)
                            # print('bbox_idx:\t', bbox_idx)
                            # print('cls:\t\t', cls_idx)
                            # print('conf:\t\t', res.boxes.conf[bbox_idx])
                            # print('ftmap:\t\t',ftmap.shape)
                            # print('ftmap_reshape:\t', ftmap.cpu().numpy().reshape(1, -1).shape)
                            # print('distance:\t', distance)
                            # print('threshold:\t', self.thresholds[cls_idx][stride_idx])

                            if self.thresholds[cls_idx][stride_idx]:
                                if distance < self.thresholds[cls_idx][stride_idx]:
                                    ood_decision[idx_img].append(1)  # InD
                                else:
                                    ood_decision[idx_img].append(0)  # OOD
                            else:
                                # logger.warning(f'WARNING: Class {cls_idx:03}, Stride {stride_idx} -> No threshold!')
                                ood_decision[idx_img].append(0)  # OOD

        else:
            raise NotImplementedError("Not implemented yet")

        return ood_decision

    def compute_ood_decision_with_ftmaps(self, activations: Union[List[np.ndarray], List[List[np.ndarray]]], bboxes: Dict[str, List], logger: Logger) -> List[List[List[int]]]:
        """
        Compute the OOD decision for each class using the in-distribution activations (usually feature maps).
        If per_class, activations must be a list of lists, where each position is a list of tensors, one for each stride.
        """
        # Como hay 3 strides y estoy con los targets (por lo que no se que clase predicha tendrian asignada),
        # voy a asignar un % de OOD a cada caja por cada stride, donde el % es el % de clases para las cuales el elemento es OOD
        # Por tanto devuelvo una lista de listas de listas, donde la primera lista es para cada imagen, la segunda para cada caja
        # y la tercera para cada stride el % de OOD
        known_classes = set(range(20))
        # Loop imagenes
        ood_decision = []
        for idx_img, activations_one_img in enumerate(activations):

            # Loop strides
            #ood_decision.append([])
            percentage_ood_one_img_all_bboxes_per_stride = []
            for idx_stride, activations_one_img_one_stride in enumerate(activations_one_img):
                #ood_decision[idx_img].append([])
                
                # Check if there is any bbox
                if len(activations_one_img_one_stride) > 0:
                    
                    # Loop clusters
                    # Loop over the cluster of each class to obtain per stride one "score" for each bbox
                    ood_decision_per_cls_per_bbox = []
                    for idx_cls, cluster in enumerate(self.clusters):
                        if len(cluster[idx_stride]) > 0:
                            distances_per_bbox = self.compute_distance(
                                cluster[idx_stride][None, :],
                                #self.activations_transformation(activations_one_img_one_stride.reshape(activations_one_img_one_stride.shape[0], -1))
                                activations_one_img_one_stride.reshape(activations_one_img_one_stride.shape[0], -1)  # Flatten the activations
                            )
                            # IN THIS CASE: 1 is OoD, 0 is InD
                            # Ya que las distancias que sean mayores que el threshold son OoD
                            ood_decision_per_cls_per_bbox.append((distances_per_bbox > self.thresholds[idx_cls][idx_stride]).astype(int))
                            # Check 
                        else:
                            if idx_cls > 19:
                                continue
                            # TODO: Aqui tengo que hacer que el numero de distancias = 1000 sea como el numero de cajas
                            raise ValueError("The clusters must have at least one sample")
                    
                    # Compute the percentage of OOD for each bbox
                    ood_decision_per_cls_per_bbox = np.stack(ood_decision_per_cls_per_bbox, axis=1)
                    percentage_ood_one_img_all_bboxes_per_stride.append(np.sum(ood_decision_per_cls_per_bbox, axis=1) / 20)

                # Check for each bbox if the cls is in the known classes and if it is, compare only agains the corresponding cluster
                for _bbox_idx, _cls_idx in enumerate(bboxes['cls'][idx_img]):
                    _cls_idx = int(_cls_idx.item())
                    if _cls_idx in known_classes:
                        # Compruebo si hay cluster
                        if len(self.clusters[_cls_idx][idx_stride]) > 0:
                            # Calculo distancia si hay cluster
                            distance = self.compute_distance(
                                self.clusters[_cls_idx][idx_stride][None, :],
                                activations_one_img_one_stride[_bbox_idx].reshape(1, -1)
                            )[0]
                            # Checkeo si la distancia es menor que el threshold, en ese caso es InD (0), sino OoD (1)
                            if self.thresholds[_cls_idx][idx_stride]:
                                if distance < self.thresholds[_cls_idx][idx_stride]:
                                    percentage_ood_one_img_all_bboxes_per_stride[idx_stride][_bbox_idx] = 0
                                else:
                                    percentage_ood_one_img_all_bboxes_per_stride[idx_stride][_bbox_idx] = 1
                            else:
                                percentage_ood_one_img_all_bboxes_per_stride[idx_stride][_bbox_idx] = 1
                        else:
                            raise ValueError("The clusters must have at least one sample")
                    else:
                        # Si la clase no es conocida, no hay que cambiar nada
                        pass
                        #percentage_ood_one_img_all_bboxes_per_stride[bbox_idx] = 0
            
            ood_decision.append(np.stack(percentage_ood_one_img_all_bboxes_per_stride, axis=1))
            print(f'{idx_img} image done!')

        return ood_decision

    def generate_clusters(self, ind_tensors: Union[List[np.ndarray], List[List[np.ndarray]]], logger: Logger) -> Union[List[np.ndarray], List[List[np.ndarray]]]:
        """
        Generate the clusters for each class using the in-distribution tensors (usually feature maps).
        If per_stride, ind_tensors must be a list of lists, where each position is
            a list of tensors, one for each stride List[List[N, C, H, W]].
            Otherwise each position is just a tensor List[[N, C, H, W]].
        """
        if self.per_class:

            if self.per_stride:

                clusters_per_class_and_stride = [[[] for _ in range(3)] for _ in range(len(ind_tensors))]

                if self.cluster_method == 'one':
                    self.generate_one_cluster_per_class_and_stride(ind_tensors, clusters_per_class_and_stride, logger)

                elif self.cluster_method in self.available_cluster_methods:
                    raise NotImplementedError("Not implemented yet")

                elif self.cluster_method == 'all':
                    raise NotImplementedError("As the amount of In-Distribution data is too big," \
                                            "ir would be intractable to treat each sample as a cluster")

                else:
                    raise NameError(f"The clustering_opt must be one of the following: 'one', 'all', or one of {self.available_cluster_methods}." \
                                    f"Current value: {self.cluster_method}")
                
            else:
                raise NotImplementedError("Not implemented yet")
            
        else:
            raise NotImplementedError("Not implemented yet")

        return clusters_per_class_and_stride
    
    def generate_one_cluster_per_class_and_stride(self, ind_tensors: List[List[np.ndarray]], clusters_per_class_and_stride: List[List[np.ndarray]], logger):
        for idx_cls, ftmaps_one_cls in enumerate(ind_tensors):

            logger.info(f'Class {idx_cls:03} of {len(ind_tensors)}')
            for idx_stride, ftmaps_one_cls_one_stride in enumerate(ftmaps_one_cls):
                
                if len(ftmaps_one_cls_one_stride) > 1:

                    #ftmaps_one_cls_one_stride = ftmaps_one_cls_one_stride.reshape(ftmaps_one_cls_one_stride.shape[0], -1)
                    ftmaps_one_cls_one_stride = self.activations_transformation(ftmaps_one_cls_one_stride)
                    clusters_per_class_and_stride[idx_cls][idx_stride] = self.agg_method(ftmaps_one_cls_one_stride, axis=0)

                    if len(ftmaps_one_cls_one_stride) < 50:
                        logger.warning(f'WARNING: Class {idx_cls:03}, Stride {idx_stride} -> Only {len(ftmaps_one_cls_one_stride)} samples')

                else:
                    logger.warning(f'SKIPPING Class {idx_cls:03}, Stride {idx_stride} -> NO SAMPLES')
                    clusters_per_class_and_stride[idx_cls][idx_stride] = np.empty(0)

    def activations_transformation(self, activations: np.array) -> np.array:
        """
        Transform the activations to the shape needed to compute the distance.
        By default, it flattens the activations leaving the batch dimension as the first dimension.
        """
        return activations.reshape(activations.shape[0], -1)


class Mahalanobis(DistanceMethod):
    # if compute_covariance:
    #     clusters_per_class_and_stride = [
    #             self.agg_method(ftmaps_one_cls_one_stride, axis=0),
    #             np.cov(ftmaps_one_cls_one_stride, rowvar=False)  # rowvar to represent variables in columns
    #     ]
    pass


class L1DistanceOneClusterPerStride(DistanceMethod):
    
        # name: str, agg_method: str, per_class: bool, per_stride: bool, cluster_method: str, cluster_optimization_metric: str
        def __init__(self, **kwargs):
            name = 'L1DistancePerStride'
            per_class = True
            per_stride = True
            cluster_method = 'one'
            cluster_optimization_metric = 'silhouette'
            super().__init__(name, per_class, per_stride, cluster_method, cluster_optimization_metric, **kwargs)
        
        def compute_distance(self, cluster: np.array, activations: np.array) -> List[float]:

            distances = pairwise_distances(
                cluster,
                activations,
                metric='l1'
                )

            return distances[0]
        

class L2DistanceOneClusterPerStride(DistanceMethod):
    
        # name: str, agg_method: str, per_class: bool, per_stride: bool, cluster_method: str, cluster_optimization_metric: str
        def __init__(self,  **kwargs):
            name = 'L2DistancePerStride'
            per_class = True
            per_stride = True
            cluster_method = 'one'
            cluster_optimization_metric = 'silhouette'
            super().__init__(name, per_class, per_stride, cluster_method, cluster_optimization_metric, **kwargs)
        
        def compute_distance(self, cluster: np.array, activations: np.array) -> List[float]:

            distances = pairwise_distances(
                cluster,
                activations,
                metric='l2'
                )

            return distances[0]


class CosineDistanceOneClusterPerStride(DistanceMethod):
    
        # name: str, agg_method: str, per_class: bool, per_stride: bool, cluster_method: str, cluster_optimization_metric: str
        #def __init__(self, agg_method, **kwargs):
        def __init__(self, **kwargs):
            name = 'CosineDistancePerStride'
            per_class = True
            per_stride = True
            cluster_method = 'one'
            cluster_optimization_metric = 'silhouette'
            #super().__init__(name, per_class, per_stride, cluster_method, cluster_optimization_metric, **kwargs)
            super().__init__(name, per_class, per_stride, cluster_method, cluster_optimization_metric, **kwargs)
        
        def compute_distance(self, cluster: np.array, activations: np.array) -> List[float]:

            distances = pairwise_distances(
                cluster,
                activations,
                metric='cosine'
                )

            return distances[0]
        

class GAPL2DistanceOneClusterPerStride(DistanceMethod):
    
        # name: str, agg_method: str, per_class: bool, per_stride: bool, cluster_method: str, cluster_optimization_metric: str
        def __init__(self,  **kwargs):
            name = 'GAP_L2DistancePerStride'
            per_class = True
            per_stride = True
            cluster_method = 'one'
            cluster_optimization_metric = 'silhouette'
            super().__init__(name, per_class, per_stride, cluster_method, cluster_optimization_metric, **kwargs)
        
        def compute_distance(self, cluster: np.array, activations: np.array) -> List[float]:

            distances = pairwise_distances(
                cluster,
                activations,
                metric='l2'
                )

            return distances[0]
        
        def activations_transformation(self, activations: np.array) -> np.array:
            """
            Transform the activations to the shape needed to compute the distance.
            By default, it flattens the activations leaving the batch dimension as the first dimension.
            """
            return np.mean(activations, axis=(2,3))  # Already reshapes to [N, features]


# Class for extracting convolutional activations from the model
class ActivationsExtractor(DistanceMethod):

    # name: str, agg_method: str, per_class: bool, per_stride: bool, cluster_method: str, cluster_optimization_metric: str
    def __init__(self,  **kwargs):
        name = 'ActivationsExtractor'
        per_class = True
        per_stride = True
        cluster_method = 'one'
        cluster_optimization_metric = 'silhouette'
        super().__init__(name, per_class, per_stride, cluster_method, cluster_optimization_metric, **kwargs)

    def compute_distance(self, cluster: np.array, activations: np.array) -> List[float]:
        raise NotImplementedError("Not implemented yet")
    
    def activations_transformation(self, activations: np.array) -> np.array:
        """
        Transform the activations to the shape needed to compute the distance.
        By default, it flattens the activations leaving the batch dimension as the first dimension.
        """
        raise NotImplementedError("Not implemented yet")
        return np.mean(activations, axis=(2,3))  # Already reshapes to [N, features]
    
    def iterate_data_to_extract_ind_activations_and_create_its_annotations(self, data_loader: InfiniteDataLoader, model, device, split: str):
        """
        Custom function to iterate over the data to extract the internal activations of the model along
        with the annotations for the dataset. They will be in the same order as in the 
        """
        if self.per_class:
            if self.per_stride:
                all_internal_activations = [[[] for _ in range(3)] for _ in range(len(model.names))]
                all_labels_for_internal_activations = [[[] for _ in range(3)] for _ in range(len(model.names))]
            else:
                all_internal_activations = [[] for _ in range(len(model.names))]

        # TODO: Cargar anotaciones del json en funcion del parametro split y REFERENCIAR A ROOT PATH con parents y demas
        tao_frames_root_path = Path('/home/tri110414/nfs_home/datasets/TAO/frames')
        with open('annotations/train.json') as f:
            data = json.load(f)

        # Obtain the bbox format from the last transform of the dataset
        if hasattr(data_loader.dataset.transforms.transforms[-1], "bbox_format"):
            bbox_format = data_loader.dataset.transforms.transforms[-1].bbox_format
        else:
            bbox_format=data_loader.dataset.labels[0]['bbox_format']

        # Start iterating over the data
        number_of_batches = len(data_loader)
        for idx_of_batch, data in enumerate(data_loader):
            
            if idx_of_batch % 50 == 0:
                print(f"{(idx_of_batch/number_of_batches) * 100:02.1f}%: Procesing batch {idx_of_batch} of {number_of_batches}")
                
            ### Prepare images and targets to feed the model ###
            imgs, targets = self.prepare_data_for_model(data, device)

            ### Process the images to get the results (bboxes and clasification) and the extra info for OOD detection ###
            results = model.predict(imgs, save=False, verbose=False)

            ### Match the predicted boxes to the ground truth boxes ###
            self.match_predicted_boxes_to_targets(results, targets, self.iou_threshold_for_matching)

            ### Extract the internal activations of the model depending on the OOD method ###
            self.extract_internal_activations(results, all_internal_activations, targets)

            # TODO: Ir recolectando los targets para poder hacer el match con las predicciones.
            #   Recolectar haciendo una lista de listas de listas similar, solo que en vez de un array
            #   por posicion va a ser un dict por cada posicion con la anotacion correspondiente

        ### Final formatting of the internal activations ###
        self.format_internal_activations(all_internal_activations)

        # TODO: Tras recolectar, hay que asignar las anotaciones a las predicciones y hacer el match pero
        #   manteniendo el orden de los videos

        return all_internal_activations
    

# Class for extracting convolutional activations from the model
class FeaturemapExtractor(DistanceMethod):

    # name: str, agg_method: str, per_class: bool, per_stride: bool, cluster_method: str, cluster_optimization_metric: str
    def __init__(self,  **kwargs):
        name = 'FeaturemapExtractor'
        per_class = True
        per_stride = True
        cluster_method = 'one'
        cluster_optimization_metric = 'silhouette'
        super().__init__(name, per_class, per_stride, cluster_method, cluster_optimization_metric, **kwargs)

    def compute_distance(self, cluster: np.array, activations: np.array) -> List[float]:
        raise NotImplementedError("Not implemented yet")
    
    def activations_transformation(self, activations: np.array) -> np.array:
        """
        Transform the activations to the shape needed to compute the distance.
        By default, it flattens the activations leaving the batch dimension as the first dimension.
        """
        raise NotImplementedError("Not implemented yet")
        return np.mean(activations, axis=(2,3))  # Already reshapes to [N, features]
    
    def iterate_data_to_extract_ind_activations_and_create_its_annotations(self, data_loader: InfiniteDataLoader, model, device, split: str):
        """
        Custom function to iterate over the data to extract the internal activations of the model along
        with the annotations for the dataset. They will be in the same order as in the 
        """
        if self.per_class:
            if self.per_stride:
                all_internal_activations = [[[] for _ in range(3)] for _ in range(len(model.names))]
                all_labels_for_internal_activations = [[[] for _ in range(3)] for _ in range(len(model.names))]
            else:
                all_internal_activations = [[] for _ in range(len(model.names))]

        # TODO: Cargar anotaciones del json en funcion del parametro split y REFERENCIAR A ROOT PATH con parents y demas
        tao_frames_root_path = Path('/home/tri110414/nfs_home/datasets/TAO/frames')
        with open('annotations/train.json') as f:
            data = json.load(f)

        # Obtain the bbox format from the last transform of the dataset
        if hasattr(data_loader.dataset.transforms.transforms[-1], "bbox_format"):
            bbox_format = data_loader.dataset.transforms.transforms[-1].bbox_format
        else:
            bbox_format=data_loader.dataset.labels[0]['bbox_format']

        # Start iterating over the data
        number_of_batches = len(data_loader)
        for idx_of_batch, data in enumerate(data_loader):
            
            if idx_of_batch % 50 == 0:
                print(f"{(idx_of_batch/number_of_batches) * 100:02.1f}%: Procesing batch {idx_of_batch} of {number_of_batches}")
                
            ### Prepare images and targets to feed the model ###
            imgs, targets = self.prepare_data_for_model(data, device)

            ### Process the images to get the results (bboxes and clasification) and the extra info for OOD detection ###
            results = model.predict(imgs, save=False, verbose=False)

            ### Match the predicted boxes to the ground truth boxes ###
            self.match_predicted_boxes_to_targets(results, targets, self.iou_threshold_for_matching)

            ### Extract the internal activations of the model depending on the OOD method ###
            self.extract_internal_activations(results, all_internal_activations, targets)

            # TODO: Ir recolectando los targets para poder hacer el match con las predicciones.
            #   Recolectar haciendo una lista de listas de listas similar, solo que en vez de un array
            #   por posicion va a ser un dict por cada posicion con la anotacion correspondiente

        ### Final formatting of the internal activations ###
        self.format_internal_activations(all_internal_activations)

        # TODO: Tras recolectar, hay que asignar las anotaciones a las predicciones y hacer el match pero
        #   manteniendo el orden de los videos

        return all_internal_activations
    

### Method to configure internals of the model ###
    
def configure_extra_output_of_the_model(model: YOLO, ood_method: Type[OODMethod]):
        # Modify the model's attributes to output the desired extra_item
        # 1. Select the layers to extract depending on the OOD method from ultralytics/nn/tasks.py
        if ood_method.which_internal_activations in FTMAPS_RELATED_OPTIONS:
            model.model.which_layers_to_extract = "convolutional_layers"
        elif ood_method.which_internal_activations == LOGITS_RELATED_OPTIONS:
            model.model.which_layers_to_extract = "logits"
        else:
            raise ValueError(f"The option {ood_method.which_internal_activations} is not valid.")
        # 2. Select the extraction mode for the ultralytics/yolo/v8/detect/predict.py
        model.model.extraction_mode = ood_method.which_internal_activations  # This attribute is created in the DetectionModel class


############################################################

# Code below copied from https://github.com/KingJamesSong/RankFeat/blob/main/utils/test_utils.py

############################################################

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level, pos_label=1.):
    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(in_examples, out_examples):
    num_in = in_examples.shape[0]
    num_out = out_examples.shape[0]

    # logger.info("# in example is: {}".format(num_in))
    # logger.info("# out example is: {}".format(num_out))

    labels = np.zeros(num_in + num_out, dtype=np.int32)
    labels[:num_in] += 1

    examples = np.squeeze(np.vstack((in_examples, out_examples)))
    aupr_in = sk.average_precision_score(labels, examples)
    auroc = sk.roc_auc_score(labels, examples)

    recall_level = 0.95
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    labels_rev = np.zeros(num_in + num_out, dtype=np.int32)
    labels_rev[num_in:] += 1
    examples = np.squeeze(-np.vstack((in_examples, out_examples)))
    aupr_out = sk.average_precision_score(labels_rev, examples)
    return auroc, aupr_in, aupr_out, fpr