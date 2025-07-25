from typing import List, Tuple, Callable, Type, Union, Dict, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
from logging import Logger
import json
import inspect
import os
import time
import datetime
from datetime import timedelta

import numpy as np
# import sklearn.metrics as sk
from numpy.core.multiarray import array as array
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
import torch
import torch.nn.functional as F
from torch import Tensor
import torchvision.ops as t_ops
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy

from ultralytics import YOLO
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import extract_roi_aligned_features_from_correct_stride
from visualization_utils import plot_results, save_image_from_results_and_data
from datasets_utils.owod.owod_evaluation_protocol import compute_metrics
from unknown_localization_utils import extract_bboxes_from_saliency_map_and_thresholds
from constants import IND_INFO_CREATION_OPTIONS, AVAILABLE_CLUSTERING_METHODS, \
    AVAILABLE_CLUSTER_OPTIMIZATION_METRICS, INTERNAL_ACTIVATIONS_EXTRACTION_OPTIONS, \
    FTMAPS_RELATED_OPTIONS, LOGITS_RELATED_OPTIONS, STRIDES_RATIO, IMAGE_FORMAT, TEMPORAL_STORAGE_PATH
from custom_hyperparams import CUSTOM_HYP
from cluster_utils import find_optimal_number_of_clusters_one_class_one_stride_and_return_labels


NOW_ood_utils = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


#@dataclass(slots=True)
class OODMethod(ABC):
    """
    Base class for all the OOD methods. It contains the basic structure of the methods and the abstract methods that
    need to be overriden by each method. It also contains some helper functions that can be used by all the methods.
    Attributes:
        name: str -> Name of the method
        is_distance_method: bool -> True if the method uses a distance to measure the OOD, False if it uses a similarity
        per_class: bool -> True if the method computes a threshold for each class, False if it computes a single threshold
        per_stride: bool -> True if the method computes a threshold for each stride, False if it computes a single threshold
        thresholds: Union[List[float], List[List[float]]] -> The thresholds for each class and stride
        iou_threshold_for_matching: float -> The threshold to use when matching the predicted boxes to the ground truth boxes
        min_conf_threshold_train: float -> Define the minimum threshold to output a box when predicting for training OOD methods
        min_conf_threshold_test: float -> Define the minimum threshold to output a box when predicting
        which_internal_activations: str -> Where to extract internal activations from.
    """
    name: str
    is_distance_method: bool
    cluster_method: str
    per_class: bool
    per_stride: bool
    thresholds: Union[List[float], List[List[float]]]
    # The threshold to use when matching the predicted boxes to the ground truth boxes.
    #   All boxes with an IoU lower than this threshold will be considered bad predictions
    iou_threshold_for_matching: float
    # Define the minimum thresholds. All boxes with a confidence lower than this threshold will be discarded automatically
    min_conf_threshold_train: float  # Train threshold, for configuring the In-Distribution
    min_conf_threshold_test: float  # Test threshold, for outputs during tests
    which_internal_activations: str  # Where to extract internal activations from
    use_values_before_sigmoid: bool  # If True, the method will use the values before the sigmoid in the logits methods
    enhanced_unk_localization: bool  # If True, the method will try to enhance the localization of the UNK objects by adding new boxes
    compute_saliency_map_one_stride: Callable  # Function to compute the saliency map of one stride
    compute_thresholds_out_of_saliency_map: Callable  # Function to compute the thresholds out of the saliency map
    #extract_bboxes_from_saliency_map_and_thresholds: Callable  # Function to extract the bboxes from the saliency map and the thresholds

    def __init__(self, name: str, is_distance_method: bool, per_class: bool, per_stride: bool, iou_threshold_for_matching: float,
                 min_conf_threshold_train: float, min_conf_threshold_test: float, which_internal_activations: str, enhanced_unk_localization: bool = False,
                 saliency_map_computation_function: Callable = None, thresholds_out_of_saliency_map_function: Callable = None, **kwargs
        ):
        self.name = name
        self.is_distance_method = is_distance_method
        self.per_class = per_class
        self.per_stride = per_stride
        self.iou_threshold_for_matching = iou_threshold_for_matching
        self.min_conf_threshold_train = min_conf_threshold_train
        self.min_conf_threshold_test = min_conf_threshold_test
        self.thresholds = None  # This will be computed later
        self.which_internal_activations = self.validate_internal_activations_option(which_internal_activations)
        self.enhanced_unk_localization = enhanced_unk_localization
        if enhanced_unk_localization:
            self.compute_saliency_map_one_stride = self.validate_saliency_map_computation_function(saliency_map_computation_function)
            self.compute_thresholds_out_of_saliency_map = self.validate_thresholds_out_of_saliency_map_function(thresholds_out_of_saliency_map_function)
            #self.extract_bboxes_from_saliency_map_and_thresholds = extract_bboxes_from_saliency_map_and_thresholds
        self.use_values_before_sigmoid = False

    @staticmethod
    def validate_internal_activations_option(selected_option: str):
        assert selected_option in INTERNAL_ACTIVATIONS_EXTRACTION_OPTIONS, f"Invalid option selected ({selected_option}) for " \
         f"internal activations extraction. Options are: {INTERNAL_ACTIVATIONS_EXTRACTION_OPTIONS}"
        return selected_option
    
    @staticmethod
    def validate_saliency_map_computation_function(passed_function: Callable) -> Callable:
        # Check that the function passed is a valid function
        assert callable(passed_function), "The passed function is not a callable"
        # Check if the function accepts only one argument
        sig = inspect.signature(passed_function)
        params = sig.parameters
        assert len(list(params.keys())) == 1, "The passed function must accept only one argument"
        # Check if the signature accepts a np.ndarray
        assert params[list(params.keys())[0]].annotation == np.ndarray, "The passed function must accept a Tensor as input"
        # Check if the function returns a np.ndarray
        assert passed_function(np.random.rand(5, 40, 40)).shape == (40, 40), "The passed function must convert (C, H, W) to (H, W)"
        return passed_function
    
    @staticmethod
    def validate_thresholds_out_of_saliency_map_function(passed_function: Callable) -> Callable:
        # Check that the function passed is a valid function
        assert callable(passed_function), "The passed function is not a callable"
        # Check if the signatures first argument accepts a np.ndarray
        sig = inspect.signature(passed_function)
        params = sig.parameters
        assert params[list(params.keys())[0]].annotation == np.ndarray, "The passed function first argument must be the saliency map and accept a Tensor or np.ndarray as input"
        # Check if the function returns a List
        assert isinstance(passed_function(np.random.rand(80, 80)), list), "The passed function must return a list with the thresholds"
        return passed_function

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

    #@abstractmethod
    def compute_INDness_scores_on_results(self, results: Results, logger) -> List[List[float]]:
        """
        Function to be overriden by each method type to compute the OOD score for each image.
        Parameters:
            results: Results -> The results of the model predictions
            logger: Logger -> The logger to print warnings or info
        Returns:
            ood_score: List[int] -> A list of lists, where the first list is for each image and the second list is for each bbox. 
                The value is the OOD score of the bbox
        """
        pass

    @abstractmethod
    def compute_scores(self, activations, *args, **kwargs) -> np.ndarray:
        """
        Function to compute the scores of the activations. Either for one box or multiple boxes.
        The function should be overriden by each method and should try to vectorize the computation as much as possible.
        """
        pass

    @abstractmethod
    def activations_transformation(self, activations: np.array, **kwargs) -> np.array:
        """
        Transform the activations to the format needed to compute the distance to the centroids. Only in DistanceMethods.
        """
        pass

    @abstractmethod
    def compute_distance(self, centroids: np.array, features: np.array) -> np.array:
        """
        Compute the distance between the centroids and the features. Only in DistanceMethods. Only in DistanceMethods.
        """
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
            
            # Save the current image with the boxes and the targets ploted
            # from torchvision.utils import draw_bounding_boxes
            # import matplotlib.pyplot as plt
            # # Plot only the bboxes
            # bboxes = res.boxes.xyxy.cpu()
            # labels = [res.names[int(_c)] for _c in res.boxes.cls.cpu()]
            # # Plot only targets
            # #bboxes = targets['bboxes'][img_idx].cpu()
            # im = draw_bounding_boxes(
            #     torch.from_numpy(res.orig_img)[img_idx].permute(2, 0, 1),
            #     bboxes,
            #     width=10,
            #     labels=labels,
            # )
            # plt.imshow(im.permute(1, 2, 0))
            # plt.axis('off')
            # plt.savefig("prueba_bboxes.png", dpi=300, bbox_inches='tight', pad_inches=0)
            # plt.close()

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

    def iterate_data_to_extract_ind_activations(self, data_loader, model: YOLO, device: str, logger: Logger):
        """
        Function to iterate over the data and extract the internal activations of the model for each image.
        The extracted activations will be stored in a list.
        """
        logger.warning(f"Using a confidence threshold of {self.min_conf_threshold_train} for training")
        if self.per_class:
            if self.per_stride:
                all_internal_activations = [[[] for _ in range(3)] for _ in range(len(model.names))]
            else:
                all_internal_activations = [[] for _ in range(len(model.names))]

        # # Obtain the bbox format from the last transform of the dataset
        # if hasattr(data_loader.dataset.transforms.transforms[-1], "bbox_format"):
        #     bbox_format = data_loader.dataset.transforms.transforms[-1].bbox_format
        # else:
        #     bbox_format=data_loader.dataset.labels[0]['bbox_format']

        # Start iterating over the data
        number_of_batches = len(data_loader)
        for idx_of_batch, data in enumerate(data_loader):
            
            self.log_every_n_batches(50, logger, idx_of_batch, number_of_batches)
                
            ### Prepare images and targets to feed the model ###
            imgs, targets = self.prepare_data_for_model(data, device)

            # Prueba: Transform images to float
            imgs = imgs.float() / 255

            ### Process the images to get the results (bboxes and clasification) and the extra info for OOD detection ###
            results = model.predict(imgs, save=False, verbose=False, conf=self.min_conf_threshold_train, device=device)

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

        logger.warning(f"Using a confidence threshold of {self.min_conf_threshold_test} for tests")
        count_of_images = 0
        number_of_images_saved = 0
        
        ### Start iterating over the data ###
        c = 0
        for idx_of_batch, data in enumerate(dataloader):
            count_of_images += len(data['im_file'])
            if idx_of_batch < 3:
                c += len(data['im_file'])
                continue

            ### Preparar imagenes y targets ###
            imgs, targets = self.prepare_data_for_model(data, device)

            # Convert the images to float
            imgs = imgs.float() / 255
            
            ### Procesar imagenes en el modelo para obtener logits y las cajas ###
            results = model.predict(imgs, save=False, verbose=True, conf=self.min_conf_threshold_test, device=device)

            if False:
                save_image_from_results_and_data(results, data, c)

            ### Comprobar si las cajas predichas son OoD ###
            ood_decision = self.compute_ood_decision_on_results(results, logger)

            ### Añadir posibles cajas desconocidas a las predicciones ###
            if self.enhanced_unk_localization:
                directory_name = f'{now}_{self.name}'
                imgs_folder_path = folder_path / directory_name
                imgs_folder_path.mkdir(exist_ok=True)
                if CUSTOM_HYP.unk.RANK_BOXES:
                    possible_unk_bboxes, ood_decision_on_unknown, distances_per_image = self.compute_extra_possible_unkwnown_bboxes_and_decision(
                        results, data, ood_decision_of_results=ood_decision, folder_path=imgs_folder_path,
                        origin_of_idx=idx_of_batch*dataloader.batch_size
                    )
                else:
                    distances_per_image = None
                    possible_unk_bboxes, ood_decision_on_unknown = self.compute_extra_possible_unkwnown_bboxes_and_decision(
                        results, data, ood_decision_of_results=ood_decision, folder_path=imgs_folder_path,
                        origin_of_idx=idx_of_batch*dataloader.batch_size
                    )
            else:
                possible_unk_bboxes = None
                ood_decision_on_unknown = None
                distances_per_image = None
                
            class_names = {k:v for k, v in model.names.items() if k < 20}
            class_names.update({80: 'UNK'})
            plot_results( 
                class_names=class_names,
                results=results,
                folder_path=folder_path,
                now=now,
                valid_preds_only=False,
                origin_of_idx=idx_of_batch*dataloader.batch_size,
                image_format=IMAGE_FORMAT,
                ood_decision=ood_decision,
                ood_method_name=self.name,
                targets=targets,
                possible_unk_boxes=possible_unk_bboxes,
                ood_decision_on_possible_unk_boxes=ood_decision_on_unknown,
                distances_unk_prop_per_image=distances_per_image,
                use_labels=True,
                original_shapes=data['ori_shape'],
                plot_gray_bands=True,
            )
            
            number_of_images_saved += len(data['im_file'])
    
    def iterate_data_to_compute_metrics(self, model: YOLO, device: str, dataloader: InfiniteDataLoader, logger: Logger, known_classes: List[int]) -> Dict[str, float]:
        dataset_name = dataloader.dataset.data.get('yaml_file', None)
        try:
            if dataset_name:
                dataset_name = Path(dataset_name).stem
                logger.info(f' *-*-*-*-*-*-*-*-*-*- Computing metrics on {dataset_name} -*-*-*-*-*-*-*-*-*-*')
            else:
                logger.info(f' *-*-*-*-*-*-*-*-*-*- Computing metrics -*-*-*-*-*-*-*-*-*-*')
        except Exception as e:
            logger.info(f' *-*-*-*-*-*-*-*-*-*- Computing metrics -*-*-*-*-*-*-*-*-*-*')

        logger.warning(f"Using a confidence threshold of {self.min_conf_threshold_test} for tests")
        number_of_images_processed = 0
        number_of_batches = len(dataloader)
        all_preds = []
        all_targets = []
        assert hasattr(dataloader.dataset, "number_of_classes"), "The dataset does not have the attribute number_of_classes to know the number of classes known in the dataset"
        class_names = list(dataloader.dataset.data['names'].values())[:dataloader.dataset.number_of_classes]
        class_names.append('unknown')
        known_classes_tensor = torch.tensor(known_classes, dtype=torch.float32)

        # For the case of benchmarks, if we want to not compute the results everytime
        TEMPORAL_STORAGE_PATH.mkdir(exist_ok=True)

        number_of_images_saved = 0
        count_of_images = 0
        for idx_of_batch, data in enumerate(dataloader):
            count_of_images += len(data['im_file'])

            if idx_of_batch % 50 == 0 or idx_of_batch == number_of_batches - 1:
                logger.info(f"{(idx_of_batch/number_of_batches) * 100:02.1f}%: Procesing batch {idx_of_batch+1} of {number_of_batches}") 

            ### Preparar imagenes y targets ###
            imgs, targets = self.prepare_data_for_model(data, device)

            # Convert the images to float
            imgs = imgs.float() / 255

            # # Write the file names of the images alongside the number of the image
            # with open(f'./imagenames.txt', 'a') as f:
            #     for i, im_file in enumerate(data['im_file']):
            #         f.write(f'{i+number_of_images_processed} - {im_file}\n')
            # number_of_images_processed += len(data['im_file'])
            # continue

            ### Procesar imagenes en el modelo para obtener logits y las cajas ###
            # In case we are in BENCHMARKS and we want to not compute the results
            if CUSTOM_HYP.BENCHMARK_MODE:
                string_for_temp_results = f'{NOW_ood_utils}_{dataset_name}_{idx_of_batch}.pt'
                path_of_temp_results = TEMPORAL_STORAGE_PATH / string_for_temp_results
                if path_of_temp_results.exists():
                    results = torch.load(path_of_temp_results)
                else:
                    results = model.predict(imgs, save=False, verbose=False, conf=self.min_conf_threshold_test, device=device)
                    torch.save(results, path_of_temp_results)
            # Regular execution
            else:
                results = model.predict(imgs, save=False, verbose=False, conf=self.min_conf_threshold_test, device=device)

            ### Comprobar si las cajas predichas son OoD ###
            ood_decision = self.compute_ood_decision_on_results(results, logger)

            ### Añadir posibles cajas desconocidas a las predicciones ###
            if self.enhanced_unk_localization:
                if CUSTOM_HYP.unk.RANK_BOXES:
                        possible_unk_bboxes, ood_decision_on_unknown, distances_per_image = self.compute_extra_possible_unkwnown_bboxes_and_decision(
                            results, data, ood_decision_of_results=ood_decision, folder_path=None,
                            origin_of_idx=idx_of_batch*dataloader.batch_size
                        )
                else:
                    distances_per_image = None
                    possible_unk_bboxes, ood_decision_on_unknown = self.compute_extra_possible_unkwnown_bboxes_and_decision(
                        results, data, ood_decision_of_results=ood_decision, folder_path=None,
                        origin_of_idx=idx_of_batch*dataloader.batch_size
                    )

            # Cada prediccion va a ser un diccionario con las siguientes claves:
            #   'img_idx': int -> Indice de la imagen
            #   'img_name': str -> Nombre del archivo de la imagen
            #   'bboxes': List[Tensor] -> Lista de tensores con las cajas predichas
            #   'cls': List[Tensor] -> Lista de tensores con las clases predichas
            #   'conf': List[Tensor] -> Lista de tensores con las confianzas de las predicciones (en yolov8 es cls)
            #   'ood_decision': List[int] -> Lista de enteros con la decision de si la caja es OoD o no
            for img_idx, res in enumerate(results):
                # Parse the ood elements as the unknown class (80)
                ood_decision_one_image = torch.tensor(ood_decision[img_idx], dtype=torch.float32)
                unknown_mask = ood_decision_one_image == 0
                bboxes_coords = res.boxes.xyxy.cpu()
                bboxes_cls = torch.where(unknown_mask, torch.tensor(80, dtype=torch.float32), res.boxes.cls.cpu())   
                # Make all the preds to be the class 80 to known the max recall possible
                #bboxes_cls = torch.tensor(80, dtype=torch.float32).repeat(len(res.boxes.cls))
                bboxes_conf = res.boxes.conf.cpu()
                # TODO: La logica de como ignorar ciertos unknowns la tenemos que idear, ya que por el momento no tiene 
                #   sentido que las propuestas no sean unknowns
                if self.enhanced_unk_localization:
                    # Add the possible unknown boxes to the predictions
                    bboxes_coords = torch.cat([bboxes_coords, possible_unk_bboxes[img_idx]], dim=0)
                    one_image_ood_decision_on_possible_unk = torch.tensor(ood_decision_on_unknown[img_idx], dtype=torch.float32)
                    assert one_image_ood_decision_on_possible_unk.sum().item() == 0.0, "Uno de los posible unknowns es considerado como known, pero como no tenemos esa logica implementada es ERROR"
                    # TODO: Por el momento simplemente lo que hago es hacer un tensor con todo clase 80 (unk). Luego tendre que ver como gestiono el hacer que acaben siendo una clase
                    cls_unk_prop = torch.tensor(80, dtype=torch.float32).repeat(len(possible_unk_bboxes[img_idx]))
                    bboxes_cls = torch.cat([bboxes_cls, cls_unk_prop], dim=0)
                    conf_unk_prop = torch.ones(len(possible_unk_bboxes[img_idx])) * 0.150001  # TODO: Pongo 0.150001 de confianza para las propuestas de unknown
                    bboxes_conf = torch.cat([bboxes_conf, conf_unk_prop], dim=0)
                all_preds.append({
                    'img_idx': number_of_images_processed + img_idx,
                    'img_name': Path(data['im_file'][img_idx]).stem,
                    'bboxes': bboxes_coords,
                    'cls': bboxes_cls,
                    'conf': bboxes_conf,
                    #'ood_decision': torch.tensor(ood_decision[img_idx], dtype=torch.float32)
                })
                    
                # Transform the classes to index 80 if they are not in the known classes
                known_mask = torch.isin(targets['cls'][img_idx], known_classes_tensor)
                transformed_target_cls = torch.where(known_mask, targets['cls'][img_idx], torch.tensor(80, dtype=torch.float32))
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
            #     image_format='{IMAGE_FORMAT}',
            #     #ood_decision=ood_decision,
            #     targets=targets,
            # )
            number_of_images_saved += len(data['im_file'])

        # All predictions collected, now compute metrics
        results_dict = compute_metrics(all_preds, all_targets, class_names, known_classes, logger)

        # Count the number of non-unknown instances and the number of unknown instances
        number_of_known_boxes = 0 
        number_of_unknown_boxes = 0
        for _target in all_targets:
            number_of_known_boxes += torch.sum(_target['cls'] != 80).item()
            number_of_unknown_boxes += torch.sum(_target['cls'] == 80).item()
        logger.info(f"Number of target known boxes: {number_of_known_boxes}")
        logger.info(f"Number of target unknown boxes: {number_of_unknown_boxes}")

        return results_dict

    def generate_thresholds(self, ind_scores: list, tpr: float, logger: Logger) -> Union[List[float], List[List[float]]]:
        """
        Generate the thresholds for each class using the in-distribution scores.
        If per_class=True, in_scores must be a list of lists,
        where each list is the list of scores for each class.
        tpr must be in the range [0, 1]
        """
        if self.is_distance_method:
            # If the method measures distance, the higher the score, the more OOD. Therefore
            # we need to get the upper bound, the tpr*100%
            used_tpr = 100*tpr
        else:            
            # As the method is a similarity method, the higher the score, the more IND. Therefore
            # we need to get the lower bound, the (1-tpr)*100%
            used_tpr = (1 - tpr)*100

        min_number_of_samples = CUSTOM_HYP.MIN_NUMBER_OF_SAMPLES_FOR_THR
        good_number_of_samples = CUSTOM_HYP.GOOD_NUM_SAMPLES

        # One threshold per class
        if self.per_class:
            
            # One threshold per stride
            if self.per_stride:

                # Per class with stride differentiation
                thresholds = [[[] for _ in range(3)] for _ in range(len(ind_scores))]
                for idx_cls, ind_scores_one_cls in enumerate(ind_scores):
                    for idx_stride, ind_scores_one_cls_one_stride in enumerate(ind_scores_one_cls):
                        if len(ind_scores_one_cls_one_stride) > min_number_of_samples:
                            thresholds[idx_cls][idx_stride] = float(np.percentile(ind_scores_one_cls_one_stride, used_tpr, method='lower'))
                            if len(ind_scores_one_cls_one_stride) < good_number_of_samples:
                                logger.warning(f"Class {idx_cls:03}, Stride {idx_stride}: has {len(ind_scores_one_cls_one_stride)} samples. The threshold may not be accurate")
                        else:
                            if idx_cls < 20:
                                logger.warning(f'Class {idx_cls:03}, Stride {idx_stride} -> Has less than {min_number_of_samples} samples. No threshold is generated')
            
            # Same threshold for all strides
            else:
                # Per class with no stride differentiation
                thresholds = [0 for _ in range(len(ind_scores))]
                for idx_cls, cl_scores in enumerate(ind_scores):
                    if len(cl_scores) > min_number_of_samples:
                        thresholds[idx_cls] = float(np.percentile(cl_scores, used_tpr, method='lower'))
                        if len(cl_scores) < good_number_of_samples:
                            logger.warning(f"Class {idx_cls}: {len(cl_scores)} samples. The threshold may not be accurate")
                    else:
                        if idx_cls < 20:
                            logger.warning(f"Class {idx_cls} has less than {min_number_of_samples} samples. No threshold is generated")

        # One threshold for all classes
        else:
            raise NotImplementedError("Not implemented yet")
        
        return thresholds
    
    ### Uknown localization methods ###
    
    def compute_extra_possible_unkwnown_bboxes_and_decision(
            self,
            results_per_image: List[Results],
            data: Dict,
            ood_decision_of_results: List[List[int]],
            folder_path: Optional[Path] = None,
            origin_of_idx: Optional[int] = 0,
        ) -> Tuple[List[Tensor], List[List[int]]]:
        """
        Compute the possible unknown bounding boxes using the feature maps of the model.
        STEPS:
            1. Select the stride with the highest resolution (8)
                1.1. Remove the padding from the feature maps
            2. Compute the saliency map out of the selected feature maps
            3. Compute the thresholds to binarize the saliency map
            4. Extract the bounding boxes from the saliency map using the thresholds
            5. Postprocess the bounding boxes. Add the padding again and then convert them to the size of the feature maps
            6. Decide wheter the unknown boxes are OoD or not
            7. Convert the boxes to original image size

        Parameters:
            results_per_image: List[Results] -> List of Results objects with the predictions of the model
            data: Dict -> Dictionary with the data of the batch
            img_batch: Optional[Tensor] -> Tensor with the images of the batch
        Returns:
            possible_unk_boxes_per_image: List[Tensor] -> List of Tensors with the possible unknown bounding boxes. Coordinates of the boxes are
                in the size of the feature maps (feature maps are paded). Each position of the list represents an image and each tensor has the shape (n_boxes, 4).
            ood_decision_on_unk_boxes_per_image: List[List[int]] -> List of numpy arrays with the decision of the possible unknown bounding boxes. 
                Each position of the list represents an image and each numpy array has the shape (n_boxes,)
        """
        assert self.which_internal_activations == 'ftmaps_and_strides', "The method needs the full feature maps and strides to compute the possible unknown bounding boxes"
        
        # Select the stride to use for the unknown localization enhancement
        # We use the stride with the highest resolution, the first one (stride 8)
        selected_stride = 0  # Stride 8
        stride_ratio = STRIDES_RATIO[selected_stride]

        # Loop over images
        possible_unk_boxes_per_image = []
        ood_decision_on_unk_boxes_per_image = []
        distances_per_image = []
        for img_idx, res in enumerate(results_per_image):
            
            #### Compute new possible unknown bounding boxes by binarizing the feature maps ####
            ### 1. Select the stride and remove padding from the feature maps
            ratio_pad = np.array(data['ratio_pad'][img_idx][1], dtype=float)  # Take the padding
            ftmaps, _ = res.extra_item
            original_img_padding_x_y = ratio_pad
            # Compute the padding in the feature map space
            ratio_pad_for_ftmaps = ratio_pad / stride_ratio
            # Get the feature map of the selected stride and unpad it
            paded_feature_maps_of_selected_stride = ftmaps[selected_stride].cpu()
            paded_ftmaps_height, paded_ftmaps_width = paded_feature_maps_of_selected_stride.shape[1:]
            unpaded_ftmaps_of_selected_stride, padding_x_y = self.select_stride_and_remove_padding_of_ftmaps(ftmaps, selected_stride, ratio_pad_for_ftmaps)
            unpaded_ftmaps_height, unpaded_ftmaps_width = unpaded_ftmaps_of_selected_stride.shape[1], unpaded_ftmaps_of_selected_stride.shape[2]

            ### 2. Compute the saliency map out of the selected feature maps
            saliency_map = self.compute_saliency_map_one_stride(unpaded_ftmaps_of_selected_stride.cpu().numpy())

            if folder_path:  # Save the saliency map and the saliency map over the original image
                USE_GREY_BANDS = False

                import matplotlib.pyplot as plt
                from skimage.transform import resize
                plt.imshow(saliency_map, cmap='viridis')
                #plt.colorbar()
                plt.axis('off')
                plt.savefig(folder_path / f'{(origin_of_idx + img_idx):03}_saliency_map.{IMAGE_FORMAT}', bbox_inches='tight', pad_inches=0)
                plt.close()
                saliency_map_plot = resize(saliency_map, (data['img'][img_idx].shape[2], data['img'][img_idx].shape[1]))
                saliency_map_plot = saliency_map_plot - saliency_map.min()
                saliency_map_plot = saliency_map_plot / saliency_map_plot.max()
                saliency_map_plot = (saliency_map_plot * 255).astype(np.uint8)
                plt.imshow(data['img'][img_idx].permute(1, 2, 0).cpu().numpy())
                plt.imshow(saliency_map_plot, cmap='viridis', alpha=0.5)
                plt.axis('off')
                plt.savefig(folder_path / f'{(origin_of_idx + img_idx):03}_saliency_map_over_image.{IMAGE_FORMAT}', bbox_inches='tight', pad_inches=0)
                plt.close()

                # Create a gray background to plot the feature maps over it
                grey_bg = (np.ones((paded_ftmaps_height, paded_ftmaps_width, 3))*144)/255

                ## Plot the feature maps for the EUL Figure
                # Plot the unpaded feature maps over a gray image with the size of the paded feature maps
                # feature_maps_folder = folder_path / f'{(origin_of_idx + img_idx):03}_ftmaps'
                # feature_maps_folder.mkdir(exist_ok=True)
                # #plt.imshow(grey_bg, cmap='gray')
                # #start_x, start_y = padding_x_y
                # for _i_ftmap, indiv_ftmap in enumerate(unpaded_ftmaps_of_selected_stride):
                #     indiv_ftmap = indiv_ftmap.cpu().numpy()
                #     if USE_GREY_BANDS:
                #         # With grey bands
                #         from matplotlib.colors import Normalize
                #         fig, ax = plt.subplots()
                #         ax.imshow(grey_bg)
                #         # Create a colormap normalization
                #         norm = Normalize(vmin=saliency_map.min(), vmax=saliency_map.max())
                #         start_x, start_y = padding_x_y
                #         ax.imshow(indiv_ftmap, cmap='viridis', norm=norm, extent=(start_x, start_x + saliency_map.shape[1], start_y + saliency_map.shape[0], start_y))
                #         # Ensure the entire 80x80 area is shown
                #         ax.set_xlim(0, grey_bg.shape[0]-1)
                #         ax.set_ylim(grey_bg.shape[1]-1, 0)
                #         ax.axis('off')
                #         plt.savefig(feature_maps_folder / f'{_i_ftmap:03}_ftmaps_over_gray_image.{IMAGE_FORMAT}', bbox_inches='tight', pad_inches=0)
                #         plt.close()
                #     else:
                #         plt.imshow(indiv_ftmap, cmap='viridis')
                #         plt.axis('off')
                #         plt.savefig(feature_maps_folder / f'{_i_ftmap:03}_ftmaps.{IMAGE_FORMAT}', bbox_inches='tight', pad_inches=0)
                #         plt.close()
                
                ### Plot the aggreated maps, the saliency map, for the EUL Figure    
                # With grey bands
                from matplotlib.colors import Normalize
                fig, ax = plt.subplots()
                ax.imshow(grey_bg)
                # Create a colormap normalization
                norm = Normalize(vmin=saliency_map.min(), vmax=saliency_map.max())
                start_x, start_y = padding_x_y
                ax.imshow(saliency_map, cmap='viridis', norm=norm, extent=(start_x, start_x + saliency_map.shape[1], start_y + saliency_map.shape[0], start_y))
                # Ensure the entire 80x80 area is shown
                ax.set_xlim(0, grey_bg.shape[0]-1)
                ax.set_ylim(grey_bg.shape[1]-1, 0)
                ax.axis('off')
                plt.savefig(folder_path / f'{(origin_of_idx + img_idx):03}_saliency_map_over_gray_image.{IMAGE_FORMAT}', bbox_inches='tight', pad_inches=0)
                plt.close()

            ### 3. Compute the thresholds to binarize the saliency map
            thresholds = self.compute_thresholds_out_of_saliency_map(saliency_map)

            if folder_path:  # Save the thresholded images in one figure
                fig, axs = plt.subplots(1, len(thresholds), figsize=(5*len(thresholds), 5))
                for idx, thr in enumerate(thresholds):
                    axs[idx].imshow(saliency_map > thr, cmap='gray')
                    axs[idx].set_title(f'Thr: {thr:.2f}')
                plt.savefig(folder_path / f'{(origin_of_idx + img_idx):03}_thresholded_saliency_map.{IMAGE_FORMAT}', bbox_inches='tight', pad_inches=0)
                plt.close()
                ### Save each threshold separately for the EUL Figure, padded with gray
                for idx, thr in enumerate(thresholds):
                    if USE_GREY_BANDS:
                        fig, ax = plt.subplots()
                        ax.imshow(grey_bg)
                        ax.imshow(saliency_map > thr, cmap='gray', extent=(start_x, start_x + saliency_map.shape[1], start_y + saliency_map.shape[0], start_y))
                        ax.set_xlim(0, grey_bg.shape[0]-1)
                        ax.set_ylim(grey_bg.shape[1]-1, 0)
                        ax.axis('off')
                        plt.savefig(folder_path / f'{(origin_of_idx + img_idx):03}_thresholded_saliency_map_thr_gray_bg_{idx:03}.{IMAGE_FORMAT}', bbox_inches='tight', pad_inches=0)
                        plt.close()
                    else:
                        plt.imshow(saliency_map > thr, cmap='gray', extent=(start_x, start_x + saliency_map.shape[1], start_y + saliency_map.shape[0], start_y))
                        plt.axis('off')
                        plt.savefig(folder_path / f'{(origin_of_idx + img_idx):03}_thresholded_saliency_map_thr_{idx:03}.{IMAGE_FORMAT}', bbox_inches='tight', pad_inches=0)
                        plt.close()
                
            ### 4. Extract the bounding boxes from the saliency map using the thresholds
            possible_unk_boxes_per_thr = extract_bboxes_from_saliency_map_and_thresholds(saliency_map, thresholds)

            if folder_path:
                ### Save for each threshold the bboxes in red over the thresholded image and over the original image
                # The bounding box comes represented as [minr, minc, maxr, maxc], which corresponds to [y_min, x_min, y_max, x_max]. 
                from matplotlib import patches
                for idx, thr in enumerate(thresholds):
                    if USE_GREY_BANDS:
                        fig, ax = plt.subplots()
                        ax.imshow(grey_bg)
                        ax.imshow(saliency_map > thr, cmap='gray', extent=(start_x, start_x + saliency_map.shape[1], start_y + saliency_map.shape[0], start_y))
                        for bbox in possible_unk_boxes_per_thr[idx]:
                            x1, y1, x2, y2 = bbox
                            x1, y1, x2, y2 = x1 + start_x, y1 + start_y, x2 + start_x, y2 + start_y
                            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='r', facecolor='none')
                            ax.add_patch(rect)
                        ax.set_xlim(0, grey_bg.shape[0]-1)
                        ax.set_ylim(grey_bg.shape[1]-1, 0)
                        ax.axis('off')
                        plt.savefig(folder_path / f'{(origin_of_idx + img_idx):03}_thresholded_saliency_map_thr_{idx:03}_with_boxes_over_gray_bg.{IMAGE_FORMAT}', bbox_inches='tight', pad_inches=0)
                        plt.close()
                    else:
                        # Boxes over the thresholded image
                        fig, ax = plt.subplots()
                        ax.imshow(saliency_map > thr, cmap='gray', extent=(start_x, start_x + saliency_map.shape[1], start_y + saliency_map.shape[0], start_y))
                        for bbox in possible_unk_boxes_per_thr[idx]:
                            x1, y1, x2, y2 = bbox
                            x1, y1, x2, y2 = x1 + start_x, y1 + start_y, x2 + start_x, y2 + start_y
                            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='r', facecolor='none')
                            ax.add_patch(rect)
                        # ax.set_xlim(0, grey_bg.shape[0]-1)
                        # ax.set_ylim(grey_bg.shape[1]-1, 0)
                        ax.axis('off')
                        plt.savefig(folder_path / f'{(origin_of_idx + img_idx):03}_thresholded_saliency_map_thr_{idx:03}_with_boxes.{IMAGE_FORMAT}', bbox_inches='tight', pad_inches=0)
                        plt.close()
                        
                        # Boxes over the original image
                        fig, ax = plt.subplots()
                        orig_img = data['img'][img_idx].permute(1, 2, 0).cpu().numpy() 
                        # Remove padding from the original image
                        #ftmaps[:, y_padding:ftmap_height-y_padding, x_padding:ftmap_width-x_padding],
                        padded_height = (640 - data['ori_shape'][img_idx][0]) // 2
                        padded_width = (640 - data['ori_shape'][img_idx][1]) // 2
                        orig_img = orig_img[padded_height:orig_img.shape[0]-padded_height, padded_width:orig_img.shape[1]-padded_width]                        
                        start_x = start_x*stride_ratio
                        start_y = start_y*stride_ratio
                        ax.imshow(orig_img, extent=(start_x, start_x + orig_img.shape[1], start_y + orig_img.shape[0], start_y))
                        #ax.imshow(saliency_map > thr, cmap='gray', extent=(start_x, start_x + saliency_map.shape[1], start_y + saliency_map.shape[0], start_y))
                        for bbox in possible_unk_boxes_per_thr[idx]:
                            x1, y1, x2, y2 = bbox
                            x1, y1, x2, y2 = x1*stride_ratio, y1*stride_ratio, x2*stride_ratio, y2*stride_ratio
                            x1, y1, x2, y2 = x1 + start_x, y1 + start_y, x2 + start_x, y2 + start_y
                            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='r', facecolor='none')
                            ax.add_patch(rect)
                        ax.axis('off')
                        plt.savefig(folder_path / f'{(origin_of_idx + img_idx):03}_thresholded_saliency_map_thr_{idx:03}_with_boxes_over_image.{IMAGE_FORMAT}', bbox_inches='tight', pad_inches=0)
                        plt.close()
                    
            ### 5. Postprocess the bounding boxes. Add the padding again and then convert them to the size of the feature maps
            unpaded_bbox_preds = res.boxes.xyxy.to('cpu')
            # x_pad_orig, y_pad_orig = ratio_pad.astype(int)
            # unpaded_bbox_preds[:, 0] = unpaded_bbox_preds[:, 0] - x_pad_orig
            # unpaded_bbox_preds[:, 1] = unpaded_bbox_preds[:, 1] - y_pad_orig
            # unpaded_bbox_preds[:, 2] = unpaded_bbox_preds[:, 2] - x_pad_orig
            # unpaded_bbox_preds[:, 3] = unpaded_bbox_preds[:, 3] - y_pad_orig
            bbox_preds_in_ftmap_size = unpaded_bbox_preds / STRIDES_RATIO[selected_stride]  # The bounding boxes are in the original image size
            fn_output = self.postprocess_unk_bboxes(
                possible_unk_boxes_per_thr,
                padding_x_y,
                unpaded_ftmaps_shape=(unpaded_ftmaps_height, unpaded_ftmaps_width),
                bbox_preds_in_ftmap_size=bbox_preds_in_ftmap_size,
                ood_decision_of_results=ood_decision_of_results[img_idx],
                paded_feature_maps=paded_feature_maps_of_selected_stride,
                selected_stride=selected_stride,
                folder_str=(folder_path / f'{(origin_of_idx + img_idx):03}').as_posix() if folder_path else None
            )
            if CUSTOM_HYP.unk.RANK_BOXES and CUSTOM_HYP.unk.USE_HEURISTICS:
                possible_unk_boxes, distances_per_proposal = fn_output
            else:
                possible_unk_boxes = fn_output

            ### 6. Decide wheter the unknown boxes are OoD or not
            ood_decision_on_unk_boxes = self.compute_ood_decision_on_possible_unk_boxes(possible_unk_boxes, paded_feature_maps_of_selected_stride)

            ### 7. Convert the boxes to original image size
            possible_unk_boxes = possible_unk_boxes * STRIDES_RATIO[selected_stride]

            # Append the possible unknown bounding boxes and the decision to diferent lists
            possible_unk_boxes_per_image.append(possible_unk_boxes)
            ood_decision_on_unk_boxes_per_image.append(ood_decision_on_unk_boxes)
            if CUSTOM_HYP.unk.RANK_BOXES and CUSTOM_HYP.unk.USE_HEURISTICS:
                distances_per_image.append(distances_per_proposal)

        if CUSTOM_HYP.unk.RANK_BOXES:
            return possible_unk_boxes_per_image, ood_decision_on_unk_boxes_per_image, distances_per_image
        else:
            return possible_unk_boxes_per_image, ood_decision_on_unk_boxes_per_image
    
    def select_stride_and_remove_padding_of_ftmaps(self, ftmaps: List[Tensor], selected_stride: int, ratio_pad_for_ftmaps: np.ndarray) -> Tuple[Tensor, Tuple[int]]:
        """
        Select the feature maps of the selected stride and remove the padding from the feature maps.
        """
        # Select the feature maps of the selected stride
        ftmaps = ftmaps[selected_stride]
        x_padding = int(ratio_pad_for_ftmaps[0])  # The padding in the x dimension is the first element
        y_padding = int(ratio_pad_for_ftmaps[1])  # The padding in the y dimension is the second element
        ftmap_height, ftmap_width = ftmaps.shape[1], ftmaps.shape[2] 
        return ftmaps[:, y_padding:ftmap_height-y_padding, x_padding:ftmap_width-x_padding], (x_padding, y_padding)

    def compute_ood_decision_on_possible_unk_boxes(self, possible_unk_boxes_one_img: Tensor, ftmaps_of_selected_stride: Tensor) -> List:
        """
        Compute the OoD decision for the possible unknown bounding boxes.
        """
        from torchvision.ops import roi_align
        # Extract the RoI Aligned feature maps of the possible unknown bounding boxes
        roi_aligned_features = roi_align(
            input=ftmaps_of_selected_stride.unsqueeze(0),
            boxes=[possible_unk_boxes_one_img.to(ftmaps_of_selected_stride.device)],
            output_size=(1,1),
            spatial_scale = 1.0,
            aligned=False,
        )

        # As theese boxes are not asociated with a prediction, we can't known which predicted class they are. 
        ood_decision = self.compute_ood_decision_on_roi_aligned_unk_boxes(roi_aligned_features)

        ood_decision = ood_decision.tolist()

        return ood_decision

    def compute_ood_decision_on_roi_aligned_unk_boxes(self, roi_aligned_features: Tensor) -> np.ndarray:
        """
        Compute the OoD decision for the RoI Aligned feature maps of the possible unknown bounding boxes.
        """
        # TODO: Por el momento dejo que todos sean UNK. Igual esta funcion la tengo que pasar
        #   como parametro tal y como lo hago con las que computan el mapa de saliencia o los thresholds
        return np.zeros(roi_aligned_features.shape[0], dtype=int)
        
    def postprocess_unk_bboxes(self, possible_unk_boxes_per_thr: List[Tensor], padding: Tuple[int], unpaded_ftmaps_shape: Tuple[int],
                               bbox_preds_in_ftmap_size: Tensor, ood_decision_of_results: List[int], paded_feature_maps: Tensor,
                               selected_stride: int, folder_str: Optional[str] = None) -> Tensor:
        """
        Postprocess the possible unknown bounding boxes.
        Parameters:
            possible_unk_boxes_per_thr: List[Tensor] -> List of numpy arrays with the bounding boxes
                for each threshold. Each numpy array has the shape (num_boxes, 4) where the columns are
                x1, y1, x2, y2.
            padding: Tuple[int] -> The padding used to remove the LetterBox padding from the feature maps.
                The first element is the padding in the x dimension and the second element is the padding in the y dimension.
            unpaded_ftmaps_shape: Tuple[int] -> The shape of the feature maps without padding (height, width).
            bbox_preds_in_ftmap_size: Tensor -> The bounding boxes in the size of the feature maps.
            paded_feature_maps: Tensor -> The feature maps of the selected stride. The shape is (1, height, width).
            selected_stride: int -> The selected stride to use for the unknown localization enhancement.
        Returns:
            possible_unk_boxes: Tensor -> The possible unknown bounding boxes. The shape is (num_boxes, 4)
        """
        all_unk_prop = []
        all_distances_per_proposal = []
        all_closes_cluster_per_proposal = []
        # TODO: Add this to constants.py
        # For small boxes removal
        min_box_size = CUSTOM_HYP.unk.MIN_BOX_SIZE
        # For big boxes removal
        unpaded_ftmap_height, unpaded_ftmap_width = unpaded_ftmaps_shape
        max_box_size_percent = CUSTOM_HYP.unk.MAX_BOX_SIZE_PERCENT  # The percentage of the feature map size that a box can take
        #####
        # Start loop of THRs
        ####
        for idx_thr in range(len(possible_unk_boxes_per_thr)):
            unk_proposals_one_thr = possible_unk_boxes_per_thr[idx_thr].clone()
            # If there are no unknown proposals, continue
            if len(unk_proposals_one_thr) == 0:
                continue
            # Add the padding to the unknown proposals
            unk_proposals_one_thr[:, 0] += padding[0]
            unk_proposals_one_thr[:, 1] += padding[1]
            unk_proposals_one_thr[:, 2] += padding[0]
            unk_proposals_one_thr[:, 3] += padding[1]
            # Obtain the width and height of the unk_proposals_one_thr
            w, h = unk_proposals_one_thr[:, 2] - unk_proposals_one_thr[:, 0], unk_proposals_one_thr[:, 3] - unk_proposals_one_thr[:, 1]
            
            #### Heuristics to remove proposals ####
            # Do we want to remove proposals using some heuristics?
            # No
            if not CUSTOM_HYP.unk.USE_HEURISTICS:  # In case we don't want to use heuristics to remove UNK proposals
                # Just add the boxes to the list
                all_unk_prop.append(unk_proposals_one_thr)
                continue
            # Yes
            ### Simple heuristics ###
            if CUSTOM_HYP.unk.USE_SIMPLE_HEURISTICS:
                # 0: Remove the unk_proposals_one_thr from the first stride?
                if idx_thr == 0 and not CUSTOM_HYP.unk.USE_FIRST_THRESHOLD:
                    print('Remove thr 0')
                    continue
                # 1º: Remove small unk_proposals_one_thr
                mask_small_boxes = (w >= min_box_size) & (h >= min_box_size)
                # 2º: Remove big unk_proposals_one_thr
                mask_big_boxes = (w < int(max_box_size_percent * unpaded_ftmap_width)) & (h < int(max_box_size_percent * unpaded_ftmap_height))
                mask = mask_small_boxes & mask_big_boxes
                unk_proposals_one_thr = unk_proposals_one_thr[mask]

            # Only use this following part if preds are available
            if len(bbox_preds_in_ftmap_size) > 0:
                
                ### Simple heuristics ###
                if CUSTOM_HYP.unk.USE_SIMPLE_HEURISTICS:
                    if CUSTOM_HYP.unk.MAX_IOU_WITH_PREDS > 0:
                        # 3º: Remove unk_proposals_one_thr with IoU > iou_thr with the predictions
                        # Compute the IoU with the predictions
                        ious = box_iou(unk_proposals_one_thr, bbox_preds_in_ftmap_size)
                        # Remove the unk_proposals_one_thr with IoU > iou_thr
                        mask = ious.max(dim=1).values < CUSTOM_HYP.unk.MAX_IOU_WITH_PREDS
                        unk_proposals_one_thr = unk_proposals_one_thr[mask]

                    # 4º: Remove unk_proposals_one_thr that have a ratio of intersection w.r.t a prediction greater than a thr
                    if CUSTOM_HYP.unk.MAX_INTERSECTION_W_PREDS:
                        if len(unk_proposals_one_thr) > 0:
                            # Calculate the intersection areas
                            inter_x1 = torch.max(unk_proposals_one_thr[:, None, 0], bbox_preds_in_ftmap_size[:, 0])
                            inter_y1 = torch.max(unk_proposals_one_thr[:, None, 1], bbox_preds_in_ftmap_size[:, 1])
                            inter_x2 = torch.min(unk_proposals_one_thr[:, None, 2], bbox_preds_in_ftmap_size[:, 2])
                            inter_y2 = torch.min(unk_proposals_one_thr[:, None, 3], bbox_preds_in_ftmap_size[:, 3])
                            inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
                            # Calculate the areas of the prediction boxes
                            pred_areas = (bbox_preds_in_ftmap_size[:, 2] - bbox_preds_in_ftmap_size[:, 0]) * \
                                        (bbox_preds_in_ftmap_size[:, 3] - bbox_preds_in_ftmap_size[:, 1])
                            # Calculate the intersection ratios
                            intersection_ratios = inter_area / pred_areas
                            # Find the max intersection ratio for each unknown proposal
                            max_intersection_ratios, _ = intersection_ratios.max(dim=1)
                            # Remove the proposals with max intersection ratio greater than 0.9. Do that by keeping the ones that are less or equal
                            unk_proposals_one_thr = unk_proposals_one_thr[max_intersection_ratios <= CUSTOM_HYP.unk.MAX_INTERSECTION_W_PREDS]

            #### Ranking boxes using diferent methods ####
            if CUSTOM_HYP.unk.RANK_BOXES:
                if len(unk_proposals_one_thr) > 0:
                    # 5º: Rank the unk_proposals_one_thr using their features (extracted from the feature maps) and comparing them
                    #   to the centroids of every known class. Then, select the unk_proposals_one_thr that are far from the centroids
                    # Extract the RoI Aligned feature maps of the possible unknown bounding boxes
                    features_per_proposal = t_ops.roi_align(
                        input=paded_feature_maps.unsqueeze(0),
                        boxes=[unk_proposals_one_thr.float()],
                        output_size=(1,1),
                        spatial_scale = 1.0,
                        aligned=False,
                    )
                    # Compute the distance of the features to the centroids of all known clusters of selected stride
                    distances_per_proposal = []
                    for idx_cls, cluster in enumerate(self.clusters):
                        if len(cluster[selected_stride]) > 0:
                            distances_one_cls_per_bbox = self.compute_distance(
                                #cluster[selected_stride][None, :],
                                cluster[selected_stride],
                                self.activations_transformation(features_per_proposal, cls_idx=idx_cls, stride_idx=selected_stride)
                                #self.activations_transformation(activations_one_img_one_stride.reshape(activations_one_img_one_stride.shape[0], -1))
                                #activations_one_img_one_stride.reshape(activations_one_img_one_stride.shape[0], -1)  # Flatten the activations
                            )
                            distances_per_proposal.append(distances_one_cls_per_bbox)
                    # Convert to array and make some operation
                    distances_per_proposal = np.array(distances_per_proposal)
                    if CUSTOM_HYP.unk.rank.RANK_BOXES_OPERATION == 'mean':
                        distances_per_proposal = distances_per_proposal.mean(axis=0)
                    elif CUSTOM_HYP.unk.rank.RANK_BOXES_OPERATION == 'max':
                        distances_per_proposal = distances_per_proposal.max(axis=0)
                    elif CUSTOM_HYP.unk.rank.RANK_BOXES_OPERATION == 'sum':
                        distances_per_proposal = distances_per_proposal.sum(axis=0)
                    elif CUSTOM_HYP.unk.rank.RANK_BOXES_OPERATION == 'min':
                        if CUSTOM_HYP.unk.rank.USE_OOD_THR_TO_REMOVE_PROPS:
                            # Retrieve the indices that would sort the array in ascending order
                            idx_sorted = np.argsort(distances_per_proposal, axis=0)
                            idx_of_closest_cluster = idx_sorted[0]  # The closest centroid
                            # Get the minimum distance for each proposal
                            distances_per_proposal = distances_per_proposal.min(axis=0)
                        else:
                            distances_per_proposal = distances_per_proposal.min(axis=0) * 100  # To compensate the low values
                    # Geometric mean
                    elif CUSTOM_HYP.unk.rank.RANK_BOXES_OPERATION == 'geometric_mean':
                        from scipy.stats import gmean
                        #if CUSTOM_HYP.unk.rank.USE_OOD_THR_TO_REMOVE_PROPS:
                        distances_per_proposal = gmean(distances_per_proposal, axis=0)
                        #distances_per_proposal = (gmean(distances_per_proposal, axis=0) - 0.02) * 1000
                    elif CUSTOM_HYP.unk.rank.RANK_BOXES_OPERATION == 'entropy':
                        # First make the distances be a probability distribution
                        distances_per_proposal = distances_per_proposal / distances_per_proposal.sum(axis=0)
                        distances_per_proposal = entropy(distances_per_proposal, axis=0)
                        #distances_per_proposal = -np.sum(distances_per_proposal * np.log2(distances_per_proposal), axis=0)
                    else:
                        raise NotImplementedError("This operation is not implemented yet")                        
            
            # Append the unk_proposals_one_thr
            all_unk_prop.append(unk_proposals_one_thr)
            if CUSTOM_HYP.unk.RANK_BOXES:
                if len(unk_proposals_one_thr) > 0:
                    all_distances_per_proposal.append(distances_per_proposal)
                    if CUSTOM_HYP.unk.rank.USE_OOD_THR_TO_REMOVE_PROPS:
                        all_closes_cluster_per_proposal.append(idx_of_closest_cluster)
        #####
        # End loop of THRs
        ####

        # Concatenate all the unk_proposals from the different thresholds -> shape = (num_boxes, 4)
        all_unk_prop = torch.cat(all_unk_prop, dim=0).float()

        #####
        # Modify proposals based on RANK
        #####
        if CUSTOM_HYP.unk.USE_HEURISTICS and CUSTOM_HYP.unk.RANK_BOXES:
            # In case we want to rank the unk proposals, we return the distances as well
            # Return a tensor with shape (num_boxes, 4) and the distances per proposal
            if len(all_distances_per_proposal) > 0:
                all_distances_per_proposal = np.concatenate(all_distances_per_proposal)
            else:
                all_distances_per_proposal = np.array([], dtype=np.float32)

            if CUSTOM_HYP.unk.rank.MAX_NUM_UNK_BOXES_PER_IMAGE > 0 and len(all_distances_per_proposal) > 0:
                if CUSTOM_HYP.unk.rank.NMS > 0:
                    # Apply NMS to the unk_proposals
                    from torchvision.ops import nms
                    # Returns the indices of the boxes that we want to keep in DESCENDING order of scores
                    if CUSTOM_HYP.unk.rank.GET_BOXES_WITH_GREATER_RANK:
                        keep = nms(all_unk_prop, torch.from_numpy(all_distances_per_proposal).float(), iou_threshold=CUSTOM_HYP.unk.rank.NMS)
                    else:
                        keep = nms(all_unk_prop, torch.from_numpy(-all_distances_per_proposal).float(), iou_threshold=CUSTOM_HYP.unk.rank.NMS)
                    #idx_sorted = keep.numpy()
                    # all_unk_prop = all_unk_prop[keep[:CUSTOM_HYP.unk.rank.MAX_NUM_UNK_BOXES_PER_IMAGE]]
                    # all_distances_per_proposal = all_distances_per_proposal[keep[:CUSTOM_HYP.unk.rank.MAX_NUM_UNK_BOXES_PER_IMAGE]]
                    all_unk_prop = all_unk_prop[keep]
                    all_distances_per_proposal = all_distances_per_proposal[keep]
                    idx_sorted = keep

                # NMS not used
                else:
                    # Select the MAX_NUM_UNK_BOXES_PER_IMAGE unk_proposals_one_thr with the highest distance
                    if CUSTOM_HYP.unk.rank.GET_BOXES_WITH_GREATER_RANK:
                        idx_sorted = np.argsort(all_distances_per_proposal)[::-1].copy()  # Order from greater to lower
                    else:
                        idx_sorted = np.argsort(all_distances_per_proposal)  # Order from lower to greater~
                        # In any case, we only keep the MAX_NUM_UNK_BOXES_PER_IMAGE of unk_proposals_one_thr
                    all_distances_per_proposal = all_distances_per_proposal[idx_sorted]
                    all_unk_prop = all_unk_prop[idx_sorted]
                    # all_distances_per_proposal = all_distances_per_proposal[idx_sorted[:CUSTOM_HYP.unk.rank.MAX_NUM_UNK_BOXES_PER_IMAGE]]
                    # all_unk_prop = all_unk_prop[idx_sorted[:CUSTOM_HYP.unk.rank.MAX_NUM_UNK_BOXES_PER_IMAGE]]
                
                if CUSTOM_HYP.unk.rank.USE_OOD_THR_TO_REMOVE_PROPS or CUSTOM_HYP.unk.rank.USE_UNK_PROPOSALS_THR:
                    if CUSTOM_HYP.unk.rank.RANK_BOXES_OPERATION == 'min':
                        closes_cluster_per_proposal = np.concatenate(all_closes_cluster_per_proposal)[idx_sorted]
                        # In case we want to remove the unk proposals that are close to a known class
                        # We want to known if the distance is lower than the threshold
                        # If the distance is lower than the threshold, we remove the proposal
                        array_thrs = np.array(self.thresholds[:20])  # The thresholds of the known classes
                        keep_thr = all_distances_per_proposal < array_thrs[closes_cluster_per_proposal, 0]
                    # elif CUSTOM_HYP.unk.rank.RANK_BOXES_OPERATION == 'mean':
                    #     keep_thr = all_distances_per_proposal < self.thresholds[80][0]
                    # elif CUSTOM_HYP.unk.rank.RANK_BOXES_OPERATION == 'geometric_mean':
                    #     keep_thr = all_distances_per_proposal < self.thresholds[80][0]
                    else:
                        keep_thr = all_distances_per_proposal < self.thresholds[80][0]
                    if isinstance(keep_thr, np.bool_):  # In case there is only one unk proposal
                        keep_thr = torch.tensor([keep_thr])
                    else:
                        # Only need to keep if there is only more than one, otherwise it gives error when trying to slice a one element array
                        all_distances_per_proposal = all_distances_per_proposal[keep_thr]
                    all_unk_prop = all_unk_prop[keep_thr]

                if len(all_unk_prop) > CUSTOM_HYP.unk.rank.MAX_NUM_UNK_BOXES_PER_IMAGE:  # In case there are more unk proposals than the limit
                    # Limit the number of proposals
                    all_distances_per_proposal = all_distances_per_proposal[:CUSTOM_HYP.unk.rank.MAX_NUM_UNK_BOXES_PER_IMAGE]
                    all_unk_prop = all_unk_prop[:CUSTOM_HYP.unk.rank.MAX_NUM_UNK_BOXES_PER_IMAGE]

            #####
            # End ranking boxes
            #####
            return all_unk_prop, all_distances_per_proposal
        ####
        # Standar return whitout ranking the unk proposals
        ####
        # Return a tensor with shape (num_boxes, 4) in case we don't want to rank the unk proposals 
        return all_unk_prop


#################################################################################
# Create classes for each method. Methods will inherit from OODMethod,
#   will override the abstract methods and also any other function that is needed.
#################################################################################

### Superclasses for methods using logits of the model ###
class LogitsMethod(OODMethod):
    
    def __init__(self, name: str, per_class: bool, per_stride: bool, iou_threshold_for_matching: float,
                 min_conf_threshold_train: float, min_conf_threshold_test: float, use_values_before_sigmoid: bool, **kwargs):
        is_distance_method = False
        which_internal_activations = 'logits'  # Always logits for these methods
        enhanced_unk_localization = False  # By default not used with logits, as feature maps are needed.
        super().__init__(name, is_distance_method, per_class, per_stride, iou_threshold_for_matching,
                         min_conf_threshold_train, min_conf_threshold_test, which_internal_activations, enhanced_unk_localization)
        self.cluster_method = 'None'
        self.use_values_before_sigmoid = use_values_before_sigmoid

    def compute_ood_decision_on_results(self, results: Results, logger: Logger) -> List[List[int]]:
        ood_decision = []  
        for idx_img, res in enumerate(results):
            ood_decision.append([])  # Every image has a list of decisions for each bbox
            for idx_bbox in range(len(res.boxes.cls)):
                cls_idx = int(res.boxes.cls[idx_bbox].cpu())
                logits = self.activations_transformation(res.extra_item[idx_bbox].cpu())
                score = self.compute_scores(logits, cls_idx)[0]
                if score < self.thresholds[cls_idx]:
                    ood_decision[idx_img].append(0)  # OOD
                else:
                    ood_decision[idx_img].append(1)  # InD

        return ood_decision

    def compute_INDness_scores_on_results(self, results: Results, logger: Logger) -> List[List[int]]:
        scores = []  
        for idx_img, res in enumerate(results):
            scores.append([])  # Every image has a list of decisions for each bbox
            for idx_bbox in range(len(res.boxes.cls)):
                cls_idx = int(res.boxes.cls[idx_bbox].cpu())
                logits = self.activations_transformation(res.extra_item[idx_bbox].cpu())
                score = self.compute_scores(logits, cls_idx)[0]
                # AQUI hay que poner una logica para que devuelva el score entre -1 y 1.
                # -1 significa que es OOD al maximo y 1 que es IND al maximo
                scores[idx_img].append(self.compute_indness(score, cls_idx))
                
        return scores

    def compute_indness(self, score: float, cls_idx: int) -> float:
        """
        Compute an score between -1 and 1. -1 means that the score is the maximum OOD score, 1 means that is the maximum IND score.
        In logits methods, it will be based on a linear function between the threshold and the minimum and maximum score.
        The maximum is 1, as a logit cannot be greater than 1. The minimum is the minimum confidence for the predictions.
        """
        if CUSTOM_HYP.fusion.LOGITS_USE_PIECEWISE_FUNCTION:

            if score > self.thresholds[cls_idx]:
                a = 1 / (self.max_score[cls_idx] - self.thresholds[cls_idx])
                b = - self.thresholds[cls_idx] / (self.max_score[cls_idx] - self.thresholds[cls_idx])
            elif score < self.thresholds[cls_idx]:
                a = - 1 / (self.min_score[cls_idx] - self.thresholds[cls_idx])
                b = self.thresholds[cls_idx] / (self.min_score[cls_idx] - self.thresholds[cls_idx])
            else:
                print("Score is equal to the threshold")
                a = 0
                b = 0

            # if score > self.thresholds[cls_idx]:
            #     a = -1 / (self.thresholds[cls_idx] - 1)
            #     b = 1 - a
            #     return a * score + b
            # elif score < self.min_conf_threshold_train:
            #     b = -1 / (1 - (self.min_conf_threshold_test/self.thresholds[cls_idx]))
            #     a = -b / self.thresholds[cls_idx]
            # else:  # score == thr
            #     print("Score is equal to the threshold")
            #     return 0
            
            indness = a * score + b
            if CUSTOM_HYP.fusion.CLIP_FUSION_SCORES:
                return max(-1, min(indness, 1))
            return indness
        
        else:
            # TODO: Puede haber varias opciones mas
            #   1. Una funcion desde el maximo score de InDness (logit=1) hasta el 0 (donde este el thr) y que la parte de OODness
            #     sea simplemente la continuacion de la recta
            #   2. Una funcion que sea una recta el maximo OODness (logit=conf_thr) y que vaya hasta el 0 (donde este el thr) y que
            #     la parte de InDness sea simplemente la continuacion de la recta
            #   3. ...
            raise NotImplementedError("Not implemented yet")

        # Check that the function is correct by plotting it 
        # import matplotlib.pyplot as plt
        # x = np.linspace(self.min_score[cls_idx], self.max_score[cls_idx], 100)
        # a = 1 / (self.max_score[cls_idx] - self.thresholds[cls_idx])
        # b = - self.thresholds[cls_idx] / (self.max_score[cls_idx] - self.thresholds[cls_idx])
        # y1 = a * x + b
        # plt.plot(x, y1, color='red')
        # a = - 1 / (self.min_score[cls_idx] - self.thresholds[cls_idx])
        # b = self.thresholds[cls_idx] / (self.min_score[cls_idx] - self.thresholds[cls_idx])
        # y2 = a * x + b
        # plt.plot(x, y2, color='blue')
        # # Plot the grid
        # plt.grid(axis='both', linestyle='--', alpha=0.5)
        # plt.savefig('AAA_indness_function.png')
        # plt.close()
    
    def extract_internal_activations(self, results: Results, all_activations: List[float], targets: Dict[str, Tensor]):
        """
        The extracted activations will be stored in the list all_activations. 
        In this case, the scores are directly computed.
        """
        for res in results:
            # Loop over the valid predictions
            for valid_idx_one_bbox in res.valid_preds:
                #cls_idx_one_bbox = res.boxes.cls[valid_idx_one_bbox].cpu().unsqueeze(0)
                cls_idx_one_bbox = int(res.boxes.cls[valid_idx_one_bbox].cpu())
                logits_one_bbox = res.extra_item[valid_idx_one_bbox].cpu()
                #all_activations[cls_idx_one_bbox].append(self.compute_score_one_bbox(logits_one_bbox, cls_idx_one_bbox))
                # Concatenate the cls_idx to the logits in the first element and the append to the list
                all_activations[cls_idx_one_bbox].append(logits_one_bbox)
                

    def format_internal_activations(self, all_activations: Union[List[float], List[List[np.ndarray]]]):
        """
        Format the internal activations of the model. In this case, the activations are already well formatted.
        """
        # Concatenate tensors of the same class
        for idx_cls in range(len(all_activations)):
            if len(all_activations[idx_cls]) > 0:
                all_activations[idx_cls] = torch.stack(all_activations[idx_cls], dim=0)
            else:
                all_activations[idx_cls] = torch.tensor([])

    def compute_scores_from_activations(self, activations: Union[List[np.ndarray], List[List[np.ndarray]]], logger: Logger):
        """
        Compute the scores for each class using the in-distribution activations (usually feature maps). They come in form of a list of ndarrays when
            per_class True and per_stride are False, where each position of the list refers to one class and the array is a tensor of shape [N, C, H, W]. 
            When is per_class and per_stride, the first list refers to classes and the second to the strides, being the arrays of the same shape as presented.
        """
        if self.per_class:
            scores = [[] for _ in range(len(activations))]
            for idx_cls in range(len(scores)):
                if len(activations[idx_cls]) > 0:
                    scores[idx_cls] = self.compute_scores(
                        self.activations_transformation(activations[idx_cls]),
                        idx_cls
                    )
                else:
                    scores[idx_cls] = np.array([], dtype=np.float32)
        else:
            raise NotImplementedError("Not implemented yet")

        self.obtain_min_max_distances(scores)
        
        return scores

    def obtain_min_max_distances(self, scores: List[List[float]]):
        """
        Obtain the minimum and maximum distances of the scores.
        """
        if self.per_class:
            self.min_score = [[] for _ in range(len(scores))]
            self.max_score = [[] for _ in range(len(scores))]
            for idx_cls in range(len(scores)):
                if len(scores[idx_cls]) > 0:
                    self.min_score[idx_cls] = np.min(scores[idx_cls])
                    self.max_score[idx_cls] = np.max(scores[idx_cls])
                else:
                    self.min_score[idx_cls] = 0.0
                    self.max_score[idx_cls] = 0.0

    def activations_transformation(self, activations: np.array, **kwargs) -> np.array:
        return activations  # We now internally do the removal of the BBOX
        
        # # NEW ultralytics. The results always carry BBOX and CLS
        # return activations[..., 4:]

        # OLD. In the OLD detect we only took the CLS when values_before_sigmoid was True
        if self.use_values_before_sigmoid:
            return activations  # In this case the activations are already the logits
        else:
            return activations[..., 4:]  # The bbox coordinates are part of the activations and must be eliminated
        #raise NotImplementedError("This method is not needed for methods using logits")

    def compute_distance(self, centroids: np.array, features: np.array) -> np.array:
        raise NotImplementedError("This method is not needed for methods using logits")


class NoMethod(LogitsMethod):

    def __init__(self, **kwargs):
        name = 'No OoD method'
        super().__init__(name, **kwargs)
    
    def compute_scores(self, logits: Tensor, cls_idx: int) -> np.ndarray:
        # Output a score of 1 for all the predictions
        if len(logits.shape) == 1:  # In case we only have one bbox
            logits = logits.unsqueeze(0)
        return np.ones(logits.shape[0])
    
    def compute_ood_decision_on_results(self, results: Results, logger: Logger) -> List[List[int]]:
        # Output a decision of 1 for all the predictions
        ood_decision = []
        for idx_img, res in enumerate(results):
            ood_decision.append([])  # Every image has a list of decisions for each bbox
            for idx_bbox in range(len(res.boxes.cls)):
                ood_decision[idx_img].append(1)  # InD
        return ood_decision
    

class MSP(LogitsMethod):

    def __init__(self, **kwargs):
        name = 'MSP'
        super().__init__(name, **kwargs)
    
    def compute_scores(self, logits: Tensor, cls_idx: int) -> np.ndarray:
        if len(logits.shape) == 1:  # In case we only have one bbox
            logits = logits.unsqueeze(0)
        return torch.nn.functional.softmax(logits, dim=1)[:, cls_idx].numpy()


class Energy(LogitsMethod):

    temper: float

    def __init__(self, temper: float, **kwargs):
        name = 'Energy'
        super().__init__(name, **kwargs)
        self.temper = temper
    
    def compute_scores(self, logits: Tensor, cls_idx: int) -> np.ndarray:
        if len(logits.shape) == 1:  # In case we only have one bbox
            logits = logits.unsqueeze(0)
        return self.temper * torch.logsumexp(logits / self.temper, dim=1).numpy()
    

class ODIN(LogitsMethod):

    temper: float

    def __init__(self, temper: float, **kwargs):
        name = 'ODIN'
        super().__init__(name, **kwargs)
        self.temper = temper
    
    def compute_scores(self, logits: Tensor, cls_idx: int) -> np.ndarray:
        if len(logits.shape) == 1:  # In case we only have one bbox
            logits = logits.unsqueeze(0)
        return torch.nn.functional.softmax(logits/self.temper, dim=1)[:, cls_idx].numpy()


class Sigmoid(LogitsMethod):

    def __init__(self, **kwargs):
        name = 'MSP'
        super().__init__(name, **kwargs)
    
    def compute_scores(self, logits: Tensor, cls_idx: int) -> np.ndarray:
        if len(logits.shape) == 1:  # In case we only have one bbox
            logits = logits.unsqueeze(0)
        if self.use_values_before_sigmoid:  # In case the values have not been processed by a sigmoid
            logits = torch.sigmoid(logits)
        logits = logits.numpy()
        assert (cls_idx == logits.argmax(axis=1)).all(), "The max logit is not the one of the predicted class"
        return logits[:, cls_idx]


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
    metric: str

    # name: str, is_distance_method: bool, per_class: bool, per_stride: bool, iou_threshold_for_matching: float, min_conf_threshold_test: float
    # def __init__(self, name: str, agg_method: str, per_class: bool, per_stride: bool, cluster_method: str,
    #              cluster_optimization_metric: str, ind_info_creation_option: str, **kwargs):
    def __init__(self, name: str, per_class: bool, per_stride: bool, cluster_method: str, metric: str,
                 cluster_optimization_metric: str, agg_method: str, ind_info_creation_option: str, which_internal_activations: str, **kwargs):
        is_distance_method = True  # Always True for distance methods
        which_internal_activations = self.validate_correct_which_internal_activations_distance_methods(which_internal_activations)
        super().__init__(name, is_distance_method, per_class, per_stride, which_internal_activations=which_internal_activations, **kwargs)
        self.metric = metric
        self.cluster_method = self.check_cluster_method_selected(cluster_method)
        self.cluster_optimization_metric = self.check_cluster_optimization_metric_selected(cluster_optimization_metric)
        self.agg_method = self.select_agg_method(agg_method)
        self.ind_info_creation_option = self.validate_correct_ind_info_creation_option(ind_info_creation_option)
    
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

    def compute_scores(self, activations: np.array, cluster: np.array) -> np.ndarray:
        return self.compute_distance(cluster, activations)

    def compute_INDness_scores_on_results(self, results: Results, logger) -> List[List[int]]:
        """
        Compute the OOD decision for each class using the in-distribution activations, 
        either feature maps along with the strides of the bounding boxes or 
        already RoI aligned feature maps.
        Parameters:
            results: List[Results] -> List of the results of the model for each image
            logger: Logger -> Logger object to log messages
        Returns:
            ood_decision: List[List[int]] -> List of decisions for each class and for each bounding box
        """
        ood_decision = []  # List of decisions for each image
        for idx_img, res in enumerate(results):
            ood_decision.append([])  # Every image has a list of decisions for each bbox
            
            # Extract the RoI aligned feature maps from the feature maps and the strides
            if self.which_internal_activations == "ftmaps_and_strides":
                ftmaps, strides = res.extra_item
                roi_aligned_ftmaps_per_stride = extract_roi_aligned_features_from_correct_stride(
                    ftmaps=[ft[None, ...] for ft in ftmaps],
                    boxes=[res.boxes.xyxy],
                    strides=[strides],
                    #img_shape=res.orig_img.shape[2:],
                    img_shape=res.orig_img.shape[1:3],
                    device=res.boxes.xyxy.device,
                    extract_all_strides=False,
                )
                # As we are processing one image only and the output is a list per image, we take the first element
                roi_aligned_ftmaps_per_stride = roi_aligned_ftmaps_per_stride[0]

            # RoI aligned feature maps are already extracted from the model
            elif self.which_internal_activations == 'roi_aligned_ftmaps':
                roi_aligned_ftmaps_per_stride = res.extra_item
            
            else:
                raise ValueError(f"The method {self.which_internal_activations} is invalid implemented yet")

            self._compute_indness_for_one_result_from_roi_aligned_feature_maps(
                idx_img=idx_img,
                one_img_bboxes_cls_idx=res.boxes.cls.cpu(),
                roi_aligned_ftmaps_one_img_per_stride=roi_aligned_ftmaps_per_stride,  # As we are processing one image only
                ood_decision=ood_decision,
                logger=logger,
            )

        return ood_decision
    
    def _compute_indness_for_one_result_from_roi_aligned_feature_maps(
            self, idx_img: int, one_img_bboxes_cls_idx: Tensor, roi_aligned_ftmaps_one_img_per_stride, ood_decision: List, logger: Logger
        ):
        """
        Compute the OOD decision for one image using the in-distribution activations (usually feature maps).
        Pipeline:
            1. Loop over the strides. Each stride is a list of bboxes and their feature maps
            2. Compute the distance between the prediction and the cluster of the predicted class
            3. Compare the distance with the threshold
        """
        # Loop each stride of the image. Select the first element of the list as we are processing one image only
        for stride_idx, (bbox_idx_in_one_stride, ftmaps) in enumerate(roi_aligned_ftmaps_one_img_per_stride):
            
            # Only enter if there are any predictions in this stride
            if len(bbox_idx_in_one_stride) > 0:
                # Each ftmap is from a bbox prediction
                for idx, ftmap in enumerate(ftmaps):
                    bbox_idx = idx
                    cls_idx = int(one_img_bboxes_cls_idx[bbox_idx])
                    ftmap = ftmap.cpu().unsqueeze(0).numpy()  # To obtain a tensor of shape [1, C, H, W]
                    # ftmap = ftmap.cpu().flatten().unsqueeze(0).numpy()
                    # [None, :] is to do the same as unsqueeze(0) but with numpy
                    # Check if there is a cluster for the class and stride
                    if len(self.clusters[cls_idx][stride_idx]) == 0:
                        logger.warning(
                            f'Image {idx_img}, bbox {bbox_idx} is viewed as an OOD.' \
                            f'It cannot be compared as there is no cluster for class {cls_idx} and stride {stride_idx}'
                        )
                        distance = 1000
                    else:
                        distance = self.compute_distance(
                            #self.clusters[cls_idx][stride_idx][None, :],
                            self.clusters[cls_idx][stride_idx],
                            self.activations_transformation(ftmap, cls_idx=cls_idx, stride_idx=stride_idx)
                        )[0]

                    ood_decision[idx_img].append(self.compute_indness(distance, cls_idx, stride_idx))


    def compute_indness(self, score: float, cls_idx: int, stride_idx: int) -> float:
        """
        Compute an score between -1 and 1. -1 means that the score is the maximum OOD score, 1 means that is the maximum IND score.
        In logits methods, it will be based on a linear function between the threshold and the minimum and maximum score.
        The maximum is 1, as a logit cannot be greater than 1. The minimum is the minimum confidence for the predictions.
        """
        if CUSTOM_HYP.fusion.DISTANCE_USE_FROM_ZERO_TO_THR:
        
            a = -1 / (self.thresholds[cls_idx][stride_idx] - 1)
            b = 1 - a

        elif CUSTOM_HYP.fusion.DISTANCE_USE_IN_DISTRIBUTION_TO_DEFINE_LIMITS:
            if self.per_class and self.per_stride:
                # Is a piecewise function
                if isinstance(self.thresholds[cls_idx], float):  # In case we have a threshold
                    if score > self.thresholds[cls_idx][stride_idx]:
                        a = -1 / (self.max_dist[cls_idx][stride_idx] - self.thresholds[cls_idx][stride_idx])
                        b = self.thresholds[cls_idx][stride_idx] / (self.max_dist[cls_idx][stride_idx] - self.thresholds[cls_idx][stride_idx])
                    elif score < self.thresholds[cls_idx][stride_idx]:
                        a = 1 / (self.min_dist[cls_idx][stride_idx] - self.thresholds[cls_idx][stride_idx])
                        b = -self.thresholds[cls_idx][stride_idx] / (self.min_dist[cls_idx][stride_idx] - self.thresholds[cls_idx][stride_idx])
                    else:
                        print("Score is equal to the threshold")
                        a = 0
                        b = 0
                else:
                    # No thresholds for this class and stride, so we cannot compute the indness
                    print(f'Class {cls_idx}, stride {stride_idx} has no thresholds. Returning -1.')
                    return -1

            else:
                raise NotImplementedError("Not implemented yet")

        indness = a * score + b
        if CUSTOM_HYP.fusion.CLIP_FUSION_SCORES:
            return max(-1, min(indness, 1))
        return indness

        # Check that the function is correct by plotting it 
        # import matplotlib.pyplot as plt
        # x = np.linspace(self.min_dist[cls_idx][stride_idx], self.max_dist[cls_idx][stride_idx], 100)
        # a = -1 / (self.max_dist[cls_idx][stride_idx] - self.thresholds[cls_idx][stride_idx])
        # b = self.thresholds[cls_idx][stride_idx] / (self.max_dist[cls_idx][stride_idx] - self.thresholds[cls_idx][stride_idx])
        # y1 = a * x + b
        # plt.plot(x, y1, color='red')
        # a = 1 / (self.min_dist[cls_idx][stride_idx] - self.thresholds[cls_idx][stride_idx])
        # b = -self.thresholds[cls_idx][stride_idx] / (self.min_dist[cls_idx][stride_idx] - self.thresholds[cls_idx][stride_idx])
        # y2 = a * x + b
        # plt.plot(x, y2, color='blue')
        # # Plot the grid
        # plt.grid(axis='both', linestyle='--', alpha=0.5)
        # plt.savefig('AAA_indness_function.png')
        # plt.close()

        # else:
        #     # TODO: Puede haber varias opciones mas
        #     #   1. Una funcion desde el maximo score de InDness (logit=1) hasta el 0 (donde este el thr) y que la parte de OODness
        #     #     sea simplemente la continuacion de la recta
        #     #   2. Una funcion que sea una recta el maximo OODness (logit=conf_thr) y que vaya hasta el 0 (donde este el thr) y que
        #     #     la parte de InDness sea simplemente la continuacion de la recta
        #     #   3. ...
        #     raise NotImplementedError("Not implemented yet")

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
                            #img_shape=results[idx_img].orig_img.shape[2:],
                            img_shape=res.orig_img.shape[1:3],
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
                        #img_shape=res.orig_img.shape[2:],
                        img_shape=res.orig_img.shape[1:3],
                        device=device
                    )
                    
                    # Add the valid roi aligned ftmaps to the list. As we only introduce one image, we need to get the only element of roi aligned ftmaps
                    self._extract_valid_preds_from_one_image_roi_aligned_ftmaps(cls_idx_one_pred, roi_aligned_ftmaps_per_stride[0], valid_preds, all_activations)

            else:
                raise ValueError("Wrong ind_info_creation_option for the selected internal activations.")

        elif self.which_internal_activations == 'ftmaps_and_strides_exact_pos':
            
            if self.ind_info_creation_option in ['all_targets_one_stride', 'all_targets_all_strides']:
                raise NotImplementedError("Not supported")
            elif self.ind_info_creation_option in ['valid_preds_one_stride', 'valid_preds_all_strides']:
                # Loop each image fo the batch
                for res in results:
                    # Extract the features from the corresponding anchor
                    self._extract_features_from_correct_stride_and_pos(res, all_activations)

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

    def _extract_features_from_correct_stride_and_pos(
            self, res: Results, all_activations: List[List[List[np.ndarray]]]
        ):
        
        # Extract all the cls idx of all the preds of one image
        cls_idx_one_pred = res.boxes.cls.cpu()
        # Extract feature maps, the stride and the valid preds info
        ftmaps, strides = res.extra_item
        valid_preds = res.valid_preds

        # Flatten the feature maps in the HxW dimensions to properly access them later
        ftmaps = [ft.reshape(ft.shape[0], -1) for ft in ftmaps]

        # Configure the strides
        s8 = res.orig_img.shape[-1] // 8
        s8 = s8 * s8
        s16 = res.orig_img.shape[-1] // 16
        s16 = (s16 * s16) + s8
        s32 = res.orig_img.shape[-1] // 32
        s32 = (s32 * s32) + s16

        # Loop each bbox of the preds of one image
        for bbox_idx, st in enumerate(strides):
            if self.ind_info_creation_option == 'valid_preds_one_stride':
                # Use only predictions that are "valid", i.e., correctly predicted and asociated univocally GT
                # and use only the stride where the bbox is predicted
                if bbox_idx in valid_preds:
                    # Obtain the predicted class
                    pred_cls = int(cls_idx_one_pred[bbox_idx].item())
                    
                    # Obtain the feature
                    if st < s8:
                        stride_idx = 0
                        feature_for_pred = ftmaps[0][:, st]
                    elif st < s16:
                        stride_idx = 1
                        st = st - s8
                        feature_for_pred = ftmaps[1][:, st]
                    elif st < s32:
                        stride_idx = 2
                        st = st - s16
                        feature_for_pred = ftmaps[2][:, st]
                    else:
                        raise ValueError(f"stride position {st} cannot be greater than stride index max {s32}")

                    # Store the feature
                    all_activations[pred_cls][stride_idx].append(feature_for_pred.cpu().numpy())
                    
            elif self.ind_info_creation_option == 'valid_preds_all_strides':
                # Use only predictions that are "valid", i.e., correctly predicted and asociated univocally GT
                # and use all the strides 
                raise NotImplementedError("Not implemented yet")
            else:
                raise ValueError("Wrong ind_info_creation_option for the selected internal activations.")


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
                #if self.cluster_method == 'one':
                self.compute_scores_clusters_per_class_and_stride(activations, scores, logger)

                self.obtain_min_max_distances(scores)
                
            else:
                raise NotImplementedError("Not implemented yet")
            
        else:
            raise NotImplementedError("Not implemented yet")
        
        return scores
    
    def obtain_min_max_distances(self, scores):
        if self.per_class and self.per_stride:
            # Check if the class and the stride have at least one sample
            self.min_dist = [[[] for _ in range(3)] for _ in range(len(scores))]
            self.max_dist = [[[] for _ in range(3)] for _ in range(len(scores))]
            for idx_cls, scores_one_cls in enumerate(scores):
                for idx_stride, scores_one_cls_one_stride in enumerate(scores_one_cls):
                    if len(scores_one_cls_one_stride) > 0:
                        self.min_dist[idx_cls][idx_stride] = np.min(scores_one_cls_one_stride)
                        self.max_dist[idx_cls][idx_stride] = np.max(scores_one_cls_one_stride)
                    else:
                        self.min_dist[idx_cls][idx_stride] = 0
                        self.max_dist[idx_cls][idx_stride] = 0
        else:
            raise NotImplementedError("Not implemented yet")

    def compute_scores_from_activations_for_unk_proposals(self, activations: Union[List[np.ndarray], List[List[np.ndarray]]], logger: Logger) -> List[float]:
        """
        Compute the scores for the unknown proposals using the in-distribution activations (usually feature maps). They come in form of a list of ndarrays when
            per_class True and per_stride are False, where each position of the list refers to one class and the array is a tensor of shape [N, C, H, W]. 
            When is per_class and per_stride, the first list refers to classes and the second to the strides, being the arrays of the same shape as presented.
        """
        scores = []
        if self.per_class:
            if self.per_stride:
                np.set_printoptions(threshold=20)
                # Compute the scores for the unknown proposals
                for idx_cls, activations_one_cls in enumerate(activations):
                    scores_one_class = []
                    activations_one_cls_one_stride = activations_one_cls[0]
                    if len(activations_one_cls_one_stride) > 0:
                        activations_one_cls_one_stride_transformed = self.activations_transformation(activations_one_cls_one_stride, cls_idx=idx_cls, stride_idx=0)
                        #logger.info(f'Class {idx_cls:03} of {len(activations)}')
                        for clusters_one_class in self.clusters:
                            cluster_one_class_first_stride = clusters_one_class[0]
                            if len(cluster_one_class_first_stride) > 0:
                                scores_one_class.append(self.compute_scores_one_class_one_stride(
                                    #cluster_one_class_first_stride[None, :],
                                    cluster_one_class_first_stride,
                                    activations_one_cls_one_stride_transformed
                                    # activations_one_cls_one_stride.reshape(activations_one_cls_one_stride.shape[0], -1)
                                ))
                        if len(activations_one_cls_one_stride) < 50:
                            if idx_cls < 20:
                                logger.warning(f'WARNING: Class {idx_cls:03}, Stride {0} -> Only {len(activations_one_cls_one_stride)} samples')
                        # En funcion de los hyperparametros decidimos que reduccion hacer
                        scores_one_class = np.array(scores_one_class)
                        if CUSTOM_HYP.unk.rank.RANK_BOXES_OPERATION == 'mean':
                            scores_one_class = np.mean(scores_one_class, axis=0)
                        elif CUSTOM_HYP.unk.rank.RANK_BOXES_OPERATION == 'geometric_mean':
                            from scipy.stats import gmean
                            scores_one_class = gmean(scores_one_class, axis=0)
                        elif CUSTOM_HYP.unk.rank.RANK_BOXES_OPERATION == 'entropy':
                            from scipy.stats import entropy
                            scores_one_class = entropy(scores_one_class, axis=0)
                        else:
                            raise NotImplementedError("Not implemented yet")
                        
                        scores.append(scores_one_class)

                    else:
                        if idx_cls < 20:
                            logger.warning(f'SKIPPING Class {idx_cls:03}, Stride {0} -> NO SAMPLES')

                scores = np.concatenate(scores)

            else:
                raise NotImplementedError("Not implemented yet")
        else:
            raise NotImplementedError("Not implemented yet")
        
        return scores
    
    def generate_unk_prop_thr(self, scores, tpr) -> None:
        if self.is_distance_method:
            # If the method measures distance, the higher the score, the more OOD. Therefore
            # we need to get the upper bound, the tpr*100%
            used_tpr = 100*tpr
        else:            
            # As the method is a similarity method, the higher the score, the more IND. Therefore
            # we need to get the lower bound, the (1-tpr)*100%
            used_tpr = (1 - tpr)*100

        if self.per_class:
            if self.per_stride:
                self.thresholds.append(
                    [float(np.percentile(scores, used_tpr, method='lower')), [], []]
                    )
                # # Plot histogram of scores
                # import matplotlib.pyplot as plt
                # plt.hist(scores, bins=100)
                # plt.savefig(f'histogram.png')
                # plt.close()
                # plt.axvline(self.thresholds[-1][0], color='r', linestyle='dashed', linewidth=2)
            else:
                raise NotImplementedError("Not implemented yet")
        else:
            raise NotImplementedError("Not implemented yet")
    
    def compute_scores_clusters_per_class_and_stride(self, activations: List[List[np.ndarray]], scores: List[List[np.ndarray]], logger):
        """
        This function has the logic of looping over the classes and strides to then call the function that computes the scores on one class and one stride.
        """
        for idx_cls, activations_one_cls in enumerate(activations):

            #logger.info(f'Class {idx_cls:03} of {len(activations)}')
            for idx_stride, activations_one_cls_one_stride in enumerate(activations_one_cls):
                
                if len(activations_one_cls_one_stride) > 0:

                    if len(self.clusters[idx_cls][idx_stride]) > 0:
                        
                        scores[idx_cls][idx_stride] = self.compute_scores_one_class_one_stride(
                            self.clusters[idx_cls][idx_stride],
                            self.activations_transformation(activations_one_cls_one_stride, cls_idx=idx_cls, stride_idx=idx_stride)
                        )

                    if len(activations_one_cls_one_stride) < 50:
                        logger.warning(f'WARNING: Class {idx_cls:03}, Stride {idx_stride} -> Only {len(activations_one_cls_one_stride)} samples')

                else:
                    if idx_cls < 20:
                        logger.warning(f'SKIPPING Class {idx_cls:03}, Stride {idx_stride} -> NO SAMPLES')
                    scores[idx_cls][idx_stride] = np.empty(0)

    def compute_scores_one_class_one_stride(self, clusters_one_cls_one_stride: np.array,  ind_activations_one_cls_one_stride: np.array) -> List[float]:
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
        Compute the OOD decision for each class using the in-distribution activations, 
        either feature maps along with the strides of the bounding boxes or 
        already RoI aligned feature maps.
        Parameters:
            results: List[Results] -> List of the results of the model for each image
            logger: Logger -> Logger object to log messages
        Returns:
            ood_decision: List[List[int]] -> List of decisions for each class and for each bounding box
        """
        ood_decision = []  # List of decisions for each image
        for idx_img, res in enumerate(results):
            ood_decision.append([])  # Every image has a list of decisions for each bbox
            
            # Extract the RoI aligned feature maps from the feature maps and the strides
            if self.which_internal_activations == "ftmaps_and_strides":
                ftmaps, strides = res.extra_item
                roi_aligned_ftmaps_per_stride = extract_roi_aligned_features_from_correct_stride(
                    ftmaps=[ft[None, ...] for ft in ftmaps],
                    boxes=[res.boxes.xyxy],
                    strides=[strides],
                    #img_shape=res.orig_img.shape[2:],  gives error in new YOLO version
                    img_shape=res.orig_img.shape[1:3],
                    device=res.boxes.xyxy.device,
                    extract_all_strides=False,
                )
                # As we are processing one image only and the output is a list per image, we take the first element
                roi_aligned_ftmaps_per_stride = roi_aligned_ftmaps_per_stride[0]

            elif self.which_internal_activations == 'ftmaps_and_strides_exact_pos':
                # Extract all the cls idx of all the preds of one image
                cls_idx_one_pred = res.boxes.cls.cpu()
                # Extract feature maps, the stride and the valid preds info
                ftmaps, strides = res.extra_item

                # Flatten the feature maps in the HxW dimensions to properly access them later
                ftmaps = [ft.reshape(ft.shape[0], -1) for ft in ftmaps]

                # Configure the strides
                s8 = res.orig_img.shape[-1] // 8
                s8 = s8 * s8
                s16 = res.orig_img.shape[-1] // 16
                s16 = (s16 * s16) + s8
                s32 = res.orig_img.shape[-1] // 32
                s32 = (s32 * s32) + s16

                # Initialize
                roi_aligned_ftmaps_per_stride = [[] for _ in range(3)]

                # Loop each bbox of the preds of one image
                for bbox_idx, st in enumerate(strides):
                    # Obtain the feature
                    if st < s8:
                        stride_idx = 0
                        feature_for_pred = ftmaps[0][:, st]
                    elif st < s16:
                        stride_idx = 1
                        st = st - s8
                        feature_for_pred = ftmaps[1][:, st]
                    elif st < s32:
                        stride_idx = 2
                        st = st - s16
                        feature_for_pred = ftmaps[2][:, st]
                    else:
                        raise ValueError(f"stride position {st} cannot be greater than stride index max {s32}")
                    roi_aligned_ftmaps_per_stride[stride_idx].append((bbox_idx, feature_for_pred))
                
                # Rearrange features to match the expected format in self._compute_ood_decision_for_one_result_from_roi_aligned_feature_maps
                for stride_idx in range(3):
                    bbox_idx_one_stride, ftmaps_one_stride = [], []
                    if len(roi_aligned_ftmaps_per_stride[stride_idx]) == 0:
                        roi_aligned_ftmaps_per_stride[stride_idx] = (torch.empty(0), torch.empty(0))
                    else:
                        for bbox_idx, ftmap in roi_aligned_ftmaps_per_stride[stride_idx]:
                            bbox_idx_one_stride.append(bbox_idx)
                            ftmaps_one_stride.append(ftmap)
                        bbox_idx_one_stride = torch.tensor(bbox_idx_one_stride, device=ftmaps_one_stride[0].device)
                        ftmaps_one_stride = torch.stack(ftmaps_one_stride, dim=0)
                        roi_aligned_ftmaps_per_stride[stride_idx] = (bbox_idx_one_stride, ftmaps_one_stride)
                
            # RoI aligned feature maps are already extracted from the model
            elif self.which_internal_activations == 'roi_aligned_ftmaps':
                roi_aligned_ftmaps_per_stride = res.extra_item
            
            else:
                raise ValueError(f"The method {self.which_internal_activations} is invalid implemented yet")

            self._compute_ood_decision_for_one_result_from_roi_aligned_feature_maps(
                idx_img=idx_img,
                one_img_bboxes_cls_idx=res.boxes.cls.cpu(),
                roi_aligned_ftmaps_one_img_per_stride=roi_aligned_ftmaps_per_stride,
                ood_decision=ood_decision,
                logger=logger,
            )

        return ood_decision

    def _compute_ood_decision_for_one_result_from_roi_aligned_feature_maps(
            self, idx_img: int, one_img_bboxes_cls_idx: Tensor, roi_aligned_ftmaps_one_img_per_stride, ood_decision: List, logger: Logger
        ):
        """
        Compute the OOD decision for one image using the in-distribution activations (usually feature maps).
        Pipeline:
            1. Loop over the strides. Each stride is a list of bboxes and their feature maps
            2. Compute the distance between the prediction and the cluster of the predicted class
            3. Compare the distance with the threshold
        """
        # Loop each stride of the image. Select the first element of the list as we are processing one image only
        for stride_idx, (bbox_idx_in_one_stride, ftmaps) in enumerate(roi_aligned_ftmaps_one_img_per_stride):
            
            # Only enter if there are any predictions in this stride
            if len(bbox_idx_in_one_stride) > 0:
                # Each ftmap is from a bbox prediction
                for idx, ftmap in enumerate(ftmaps):
                    bbox_idx = idx
                    cls_idx = int(one_img_bboxes_cls_idx[bbox_idx])
                    ftmap = ftmap.cpu().unsqueeze(0).numpy()  # To obtain a tensor of shape [1, C, H, W]
                    # ftmap = ftmap.cpu().flatten().unsqueeze(0).numpy()
                    # [None, :] is to do the same as unsqueeze(0) but with numpy
                    # Check if there is a cluster for the class and stride
                    if len(self.clusters[cls_idx][stride_idx]) == 0:
                        logger.warning(
                            f'Image {idx_img}, bbox {bbox_idx} is viewed as an OOD.' \
                            f'It cannot be compared as there is no cluster for class {cls_idx} and stride {stride_idx}'
                        )
                        distance = 1000
                    else:
                        distance = self.compute_distance(
                            #self.clusters[cls_idx][stride_idx][None, :],
                            self.clusters[cls_idx][stride_idx],
                            self.activations_transformation(ftmap, cls_idx=cls_idx, stride_idx=stride_idx)
                        )[0]

                    # Check if the distance is lower than the threshold
                    if self.thresholds[cls_idx][stride_idx]:
                        if distance < self.thresholds[cls_idx][stride_idx]:
                            ood_decision[idx_img].append(1)  # InD
                        else:
                            ood_decision[idx_img].append(0)  # OOD
                    else:
                        # logger.warning(f'WARNING: Class {cls_idx:03}, Stride {stride_idx} -> No threshold!')
                        ood_decision[idx_img].append(0)  # OOD   


    # def compute_ood_decision_with_ftmaps(self, activations: Union[List[np.ndarray], List[List[np.ndarray]]], bboxes: Dict[str, List], logger: Logger) -> List[List[List[int]]]:
    #     """
    #     Compute the OOD decision for each class using the in-distribution activations (usually feature maps).
    #     If per_class, activations must be a list of lists, where each position is a list of tensors, one for each stride.
    #     """
    #     # Como hay 3 strides y estoy con los targets (por lo que no se que clase predicha tendrian asignada),
    #     # voy a asignar un % de OOD a cada caja por cada stride, donde el % es el % de clases para las cuales el elemento es OOD
    #     # Por tanto devuelvo una lista de listas de listas, donde la primera lista es para cada imagen, la segunda para cada caja
    #     # y la tercera para cada stride el % de OOD
    #     known_classes = set(range(20))
    #     # Loop imagenes
    #     ood_decision = []
    #     for idx_img, activations_one_img in enumerate(activations):

    #         # Loop strides
    #         #ood_decision.append([])
    #         percentage_ood_one_img_all_bboxes_per_stride = []
    #         for idx_stride, activations_one_img_one_stride in enumerate(activations_one_img):
    #             #ood_decision[idx_img].append([])
                
    #             # Check if there is any bbox
    #             if len(activations_one_img_one_stride) > 0:
                    
    #                 # Loop clusters
    #                 # Loop over the cluster of each class to obtain per stride one "score" for each bbox
    #                 ood_decision_per_cls_per_bbox = []
    #                 for idx_cls, cluster in enumerate(self.clusters):
    #                     if len(cluster[idx_stride]) > 0:
    #                         distances_per_bbox = self.compute_distance(
    #                             #cluster[idx_stride][None, :],
    #                             cluster[idx_stride],
    #                             #self.activations_transformation(activations_one_img_one_stride.reshape(activations_one_img_one_stride.shape[0], -1))
    #                             activations_one_img_one_stride.reshape(activations_one_img_one_stride.shape[0], -1)  # Flatten the activations
    #                         )
    #                         # IN THIS CASE: 1 is OoD, 0 is InD
    #                         # Ya que las distancias que sean mayores que el threshold son OoD
    #                         ood_decision_per_cls_per_bbox.append((distances_per_bbox > self.thresholds[idx_cls][idx_stride]).astype(int))
    #                         # Check 
    #                     else:
    #                         if idx_cls > 19:
    #                             continue
    #                         # TODO: Aqui tengo que hacer que el numero de distancias = 1000 sea como el numero de cajas
    #                         raise ValueError("The clusters must have at least one sample")
                    
    #                 # Compute the percentage of OOD for each bbox
    #                 ood_decision_per_cls_per_bbox = np.stack(ood_decision_per_cls_per_bbox, axis=1)
    #                 percentage_ood_one_img_all_bboxes_per_stride.append(np.sum(ood_decision_per_cls_per_bbox, axis=1) / 20)

    #             # Check for each bbox if the cls is in the known classes and if it is, compare only agains the corresponding cluster
    #             for _bbox_idx, _cls_idx in enumerate(bboxes['cls'][idx_img]):
    #                 _cls_idx = int(_cls_idx.item())
    #                 if _cls_idx in known_classes:
    #                     # Compruebo si hay cluster
    #                     if len(self.clusters[_cls_idx][idx_stride]) > 0:
    #                         # Calculo distancia si hay cluster
    #                         distance = self.compute_distance(
    #                             #self.clusters[_cls_idx][idx_stride][None, :],
    #                             self.clusters[_cls_idx][idx_stride],
    #                             activations_one_img_one_stride[_bbox_idx].reshape(1, -1)
    #                         )[0]
    #                         # Checkeo si la distancia es menor que el threshold, en ese caso es InD (0), sino OoD (1)
    #                         if self.thresholds[_cls_idx][idx_stride]:
    #                             if distance < self.thresholds[_cls_idx][idx_stride]:
    #                                 percentage_ood_one_img_all_bboxes_per_stride[idx_stride][_bbox_idx] = 0
    #                             else:
    #                                 percentage_ood_one_img_all_bboxes_per_stride[idx_stride][_bbox_idx] = 1
    #                         else:
    #                             percentage_ood_one_img_all_bboxes_per_stride[idx_stride][_bbox_idx] = 1
    #                     else:
    #                         raise ValueError("The clusters must have at least one sample")
    #                 else:
    #                     # Si la clase no es conocida, no hay que cambiar nada
    #                     pass
    #                     #percentage_ood_one_img_all_bboxes_per_stride[bbox_idx] = 0
            
    #         ood_decision.append(np.stack(percentage_ood_one_img_all_bboxes_per_stride, axis=1))
    #         print(f'{idx_img} image done!')

    #     return ood_decision

    def generate_clusters(self, ind_tensors: Union[List[np.ndarray], List[List[np.ndarray]]], logger: Logger) -> Union[List[np.ndarray], List[List[np.ndarray]]]:
        """
        Generate the clusters for each class using the in-distribution tensors (usually feature maps).
        If per_stride, ind_tensors must be a list of lists, where each position is
            a list of tensors, one for each stride List[List[N, C, H, W]].
            Otherwise each position is just a tensor List[[N, C, H, W]].
        """
        t1 = time.perf_counter()
        if self.per_class:

            if self.per_stride:

                clusters_per_class_and_stride = [[[] for _ in range(3)] for _ in range(len(ind_tensors))]

                if self.cluster_method == 'one':
                    self.generate_one_cluster_per_class_and_stride(ind_tensors, clusters_per_class_and_stride, logger)

                elif self.cluster_method in AVAILABLE_CLUSTERING_METHODS:
                    self.generate_multiple_cluster_per_class_per_stride(ind_tensors, clusters_per_class_and_stride, logger)

                else:
                    raise NameError(f"The clustering_opt must be one of the following: {AVAILABLE_CLUSTERING_METHODS}." \
                                    f"Current value: {self.cluster_method}")
                
            else:
                raise NotImplementedError("Not implemented yet")
            
        else:
            raise NotImplementedError("Not implemented yet")

        x = str(timedelta(seconds=time.perf_counter() - t1)).split(':')
        logger.info(f'Clusters generated in {x[0]} Hours, {x[1]} Minutes {x[2]} Seconds')
        return clusters_per_class_and_stride
    
    def generate_one_cluster_per_class_and_stride(self, ind_tensors: List[List[np.ndarray]], clusters_per_class_and_stride: List[List[np.ndarray]], logger):
        for idx_cls, ftmaps_one_cls in enumerate(ind_tensors):

            #logger.info(f'Class {idx_cls:03} of {len(ind_tensors)}')
            for idx_stride, ftmaps_one_cls_one_stride in enumerate(ftmaps_one_cls):
                
                if len(ftmaps_one_cls_one_stride) > CUSTOM_HYP.clusters.MIN_SAMPLES:
                    ftmaps_one_cls_one_stride = self.activations_transformation(ftmaps_one_cls_one_stride, cls_idx=idx_cls, stride_idx=idx_stride)
                    #clusters_per_class_and_stride[idx_cls][idx_stride] = self.agg_method(ftmaps_one_cls_one_stride, axis=0)
                    clusters_per_class_and_stride[idx_cls][idx_stride] = self.agg_method(ftmaps_one_cls_one_stride, axis=0)[None, :]

                    if len(ftmaps_one_cls_one_stride) < 50:
                        logger.warning(f'WARNING: Class {idx_cls:03}, Stride {idx_stride} -> Only {len(ftmaps_one_cls_one_stride)} samples')

                else:
                    if idx_cls < 20:
                        logger.warning(f'SKIPPING Class {idx_cls:03}, Stride {idx_stride} -> NO SAMPLES')
                    clusters_per_class_and_stride[idx_cls][idx_stride] = np.empty(0)

    def generate_multiple_cluster_per_class_per_stride(
            self, ind_tensors: List[List[np.ndarray]], clusters_per_class_and_stride: List[List[np.ndarray]], logger: Logger
        ):
        np.set_printoptions(threshold=20)

        if CUSTOM_HYP.clusters.VISUALIZE and not self.cluster_method == 'all':
            import matplotlib.pyplot as plt
            from collections import Counter
            folder_for_hist = Path('figures/histograms')
            folder_for_hist.mkdir(parents=False, exist_ok=True)
            folder_for_cluster_scores = Path('figures/cluster_scores')
            folder_for_cluster_scores.mkdir(parents=False, exist_ok=True)

        for idx_cls, ftmaps_one_cls in enumerate(ind_tensors):
            #logger.info(f'Class {idx_cls:03} of {len(ind_tensors)}')
            # Log when every 25% is completed
            if (idx_cls+1) % (20 // 4) == 0:
                logger.info(f'*** {idx_cls+1}/20 classes done ***')
            cluster_labels_one_class = []
            for idx_stride, ftmaps_one_cls_one_stride in enumerate(ftmaps_one_cls):
                
                if len(ftmaps_one_cls_one_stride) > CUSTOM_HYP.clusters.MIN_SAMPLES:
                    ftmaps_one_cls_one_stride = self.activations_transformation(ftmaps_one_cls_one_stride, cls_idx=idx_cls, stride_idx=idx_stride)
                    # 1. Find the optimal number of clusters and obtain the labels
                    if CUSTOM_HYP.clusters.VISUALIZE and not self.cluster_method == 'all':
                        string_for_visualization = f'{self.name}_class{idx_cls:03}_stride{idx_stride}'
                        string_for_visualization = (folder_for_cluster_scores / string_for_visualization).as_posix()
                    else:
                        string_for_visualization = ''
                    cluster_labels = find_optimal_number_of_clusters_one_class_one_stride_and_return_labels(
                        ftmaps_one_cls_one_stride,
                        self.cluster_method,
                        self.metric,
                        self.cluster_optimization_metric,
                        string_for_visualization,
                        logger,
                        visualize=CUSTOM_HYP.clusters.VISUALIZE,
                    )
                    
                    # Obtain the indices of the clusters
                    cluster_indices = set(cluster_labels)
                    cluster_labels_one_class.append(cluster_labels)

                    # 2. Aggregate the samples of each cluster using the agg method
                    clusters_centroids = []
                    for idx_cluster in sorted(cluster_indices):
                        if idx_cluster == -1 and CUSTOM_HYP.clusters.REMOVE_ORPHANS:  # To remove the samples that are not assigned to any cluster
                            logger.warning(f'WARNING: Class {idx_cls:03}, Stride {idx_stride} -> Removing {np.sum(cluster_labels == idx_cluster)} orphan samples')
                        else:
                            clusters_centroids.append(self.agg_method(ftmaps_one_cls_one_stride[cluster_labels == idx_cluster], axis=0))
                    clusters_per_class_and_stride[idx_cls][idx_stride] = np.array(clusters_centroids)

                else:
                    if idx_cls < 20:
                        logger.warning(f'SKIPPING Class {idx_cls:03}, Stride {idx_stride} -> NO SAMPLES')
                    clusters_per_class_and_stride[idx_cls][idx_stride] = np.empty(0)
                    cluster_labels_one_class.append([])

            if CUSTOM_HYP.clusters.VISUALIZE and not self.cluster_method == 'all':
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                #fig.suptitle(f'Class {idx_cls:03}')
                class_has_at_least_one_stride_with_clusters = False
                for _i, cl_lbl in enumerate(cluster_labels_one_class):
                    # Count the number of samples per cluster
                    cluster_counts = Counter(cl_lbl)
                    # Extracting cluster indices and their respective counts
                    one_stride_labels = list(cluster_counts.keys())
                    one_stride_counts = list(cluster_counts.values())
                    if len(cl_lbl) > 0:
                        class_has_at_least_one_stride_with_clusters = True
                        bars = ax[_i].bar(one_stride_labels, one_stride_counts, color='skyblue')
                        ax[_i].set_title(f'Class {idx_cls:03} - Stride {_i}')
                        ax[_i].set_xlabel('Cluster index')
                        ax[_i].set_ylabel('Number of samples')
                        ax[_i].set_xticks(one_stride_labels)  # Ensuring the x-ticks correspond to cluster indices
                        ax[_i].grid(axis='y', linestyle='--', alpha=0.7)
                        # Adding value labels on top of the bars
                        for bar in bars:
                            yval = bar.get_height()
                            ax[_i].text(bar.get_x() + bar.get_width() / 2, yval + 1, int(yval), ha='center', va='bottom')

                if class_has_at_least_one_stride_with_clusters:
                    plt.tight_layout()
                    plt.savefig(folder_for_hist / f'histogram_{self.name}_{self.cluster_method}_{idx_cls:03}_{self.cluster_optimization_metric}.png')
                    plt.close()
                else:
                    plt.close()

    def activations_transformation(self, activations: np.array, **kwargs) -> np.array:
        """
        Transform the activations to the shape needed to compute the distance.
        By default, it flattens the activations leaving the batch dimension as the first dimension.
        """
        return normalize(activations.reshape(activations.shape[0], -1), axis=1)
        #return activations.reshape(activations.shape[0], -1)


class _PairwiseDistanceClustersPerClassPerStride(DistanceMethod):
    def __init__(self, name: str, metric: str, **kwargs):
            AVAILABLE_PAIRWISE_METRICS = ['cosine', 'l1', 'l2', 'manhattan','euclidean']
            per_class = True
            per_stride = True
            super().__init__(name=name, metric=metric, per_class=per_class, per_stride=per_stride, **kwargs)
            assert self.per_class and self.per_stride, "This method is only compatible with per_class and per_stride"
            assert self.metric in AVAILABLE_PAIRWISE_METRICS, f"The metric must be one of {AVAILABLE_PAIRWISE_METRICS}. Current value: {self.metric}"
        
    def compute_distance(self, cluster: np.array, activations: np.array) -> List[float]:

        distances = pairwise_distances(
            cluster,
            activations,
            metric=self.metric
            )

        return distances.min(axis=0)


class _DimensionalityReductionMethod(_PairwiseDistanceClustersPerClassPerStride):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_dimensionality_reduction_trained = False

    @abstractmethod
    def train_dimensionality_reduction_module(self, activations: List[np.ndarray], logger: Logger):
        pass

    @abstractmethod
    def activations_transformation(self, activations: np.array, cls_idx: int, stride_idx: int) -> np.array:

        # Transform the activations to the new space by using the dimensionality reduction function
        pass
    
    # Overriding the method to train the dimensionality reduction module before generating the clusters
    def generate_clusters(self, ind_tensors: List[np.ndarray] | List[List[np.ndarray]], logger: Logger) -> List[np.ndarray] | List[List[np.ndarray]]:
        
        if not self.is_dimensionality_reduction_trained:
            self.train_dimensionality_reduction_module(ind_tensors, logger)
            self.is_dimensionality_reduction_trained = True

        return super().generate_clusters(ind_tensors, logger)

class UmapMethod(_DimensionalityReductionMethod):
    
    def __init__(self, **kwargs):
        import umap 
        per_stride = True
        if per_stride:
            kwargs = {
                'n_components': 32,
                'n_neighbors': 15,
                'min_dist': 0.1,
                'metric': 'cosine',
            }
            self.umap = [
                umap.UMAP(n_components=32, n_neighbors=15, min_dist=0.1, metric='cosine'), # random_state=42),
                umap.UMAP(n_components=32, n_neighbors=15, min_dist=0.1, metric='cosine'),
                umap.UMAP(n_components=32, n_neighbors=15, min_dist=0.1, metric='cosine')
                ]
        metric = 'cosine'
        name = 'CosineDistancePerStride'
        super().__init__(name=name, metric=metric, **kwargs)

    def train_dimensionality_reduction_module(self, activations: List[np.ndarray], logger: Logger):
        
        if self.per_class and self.per_stride:
            # Obtain the concatenated arrays of each stride, and apply a UMAP per stride
            concatenated_arrays_per_stride = [[] for _ in range(3)]
            for idx_cls, activations_one_cls in enumerate(activations):
                for idx_stride, activations_one_cls_one_stride in enumerate(activations_one_cls):
                    if len(activations_one_cls_one_stride) > 0:
                        concatenated_arrays_per_stride[idx_stride].append(activations_one_cls_one_stride)

            concatenated_arrays_per_stride = [np.concatenate(arr, axis=0) for arr in concatenated_arrays_per_stride]

            for idx_stride, concatenated_array in enumerate(concatenated_arrays_per_stride):
                self.umap[idx_stride].fit(concatenated_array.reshape(concatenated_array.shape[0], -1))
                logger.info(f'UMAP fitted for stride {idx_stride}')

    def activations_transformation(self, activations: np.array, cls_idx: int, stride_idx: int) -> np.array:
        # Depending on the stride, apply the corresponding UMAP
        return self.umap[stride_idx].transform(activations.reshape(activations.shape[0], -1))


#class IvisMethodCosinePerClusterPerStride(_DimensionalityReductionMethod):
class _IvisMethodPairwiseDistance(_DimensionalityReductionMethod):
    
    def __init__(self, metric, name, **kwargs):
        import ivis
        per_stride = True
        if per_stride:
            dims_reduction_kwargs = {
                'embedding_dims': CUSTOM_HYP.dr.ivis.EMBEDDING_DIMS,
                'n_epochs_without_progress': CUSTOM_HYP.dr.ivis.N_EPOCHS_WITHOUT_PROGRESS,
                'k': CUSTOM_HYP.dr.ivis.K,
                'model': CUSTOM_HYP.dr.ivis.MODEL,
                'distance': metric, 
            }
            self.ivis = [
                ivis.Ivis(**dims_reduction_kwargs),
                ivis.Ivis(**dims_reduction_kwargs),
                ivis.Ivis(**dims_reduction_kwargs),
                ]
        super().__init__(name=name, metric=metric, **kwargs)

    def train_dimensionality_reduction_module(self, activations: List[np.ndarray], logger: Logger):
        from sklearn.utils import shuffle

        if self.per_class and self.per_stride:
            # Obtain the concatenated arrays of each stride, and apply a UMAP per stride
            concatenated_arrays_per_stride = [[] for _ in range(3)]
            concatenated_targets_per_stride = [[] for _ in range(3)]
            for idx_cls, activations_one_cls in enumerate(activations):
                for idx_stride, activations_one_cls_one_stride in enumerate(activations_one_cls):
                    if len(activations_one_cls_one_stride) > 0:
                        concatenated_arrays_per_stride[idx_stride].append(activations_one_cls_one_stride)
                        concatenated_targets_per_stride[idx_stride].append(np.ones(activations_one_cls_one_stride.shape[0]) * idx_cls)

            concatenated_arrays_per_stride = [np.concatenate(arr, axis=0) for arr in concatenated_arrays_per_stride]
            concatenated_targets_per_stride = [np.concatenate(arr, axis=0) for arr in concatenated_targets_per_stride]

            for idx_stride, concatenated_array in enumerate(concatenated_arrays_per_stride):
                concatenated_array, concatenated_targets  = shuffle(concatenated_array.reshape(concatenated_array.shape[0], -1), concatenated_targets_per_stride[idx_stride].astype(np.uint8), random_state=15)
                self.ivis[idx_stride].fit(normalize(concatenated_array), concatenated_targets, shuffle_mode=True)
                logger.info(f'IVIS fitted for stride {idx_stride}')

    def activations_transformation(self, activations: np.array, cls_idx: int, stride_idx: int) -> np.array:
        # Depending on the stride, apply the corresponding dimensionality reduction
        return self.ivis[stride_idx].transform(
            normalize(
                activations.reshape(activations.shape[0], -1)
            )
        )

class IvisMethodCosine(_IvisMethodPairwiseDistance):
    
    def __init__(self, **kwargs):
        metric = 'cosine'
        name = 'IvisCosineDistancePerStride'
        super().__init__(metric, name, **kwargs)


class IvisMethodL1(_IvisMethodPairwiseDistance):
        
    def __init__(self, **kwargs):
        metric = 'manhattan'
        name = 'IvisL1DistancePerStride'
        super().__init__(metric, name, **kwargs)


class IvisMethodL2(_IvisMethodPairwiseDistance):

    def __init__(self, **kwargs):
        metric = 'euclidean'
        name = 'IvisL2DistancePerStride'
        super().__init__(metric, name, **kwargs)


class L1DistanceOneClusterPerStride(_PairwiseDistanceClustersPerClassPerStride):

    def __init__(self, **kwargs):
            metric = 'l1'
            name = 'L1DistancePerStride'
            super().__init__(name, metric, **kwargs)


class L2DistanceOneClusterPerStride(_PairwiseDistanceClustersPerClassPerStride):

    def __init__(self, **kwargs):
            metric = 'l2'
            name = 'L2DistancePerStride'
            super().__init__(name, metric, **kwargs)


class CosineDistanceOneClusterPerStride(_PairwiseDistanceClustersPerClassPerStride):

    def __init__(self, **kwargs):
            metric = 'cosine'
            name = 'CosineDistancePerStride'
            super().__init__(name, metric, **kwargs)


# Class for extracting convolutional activations from the model
class ActivationsExtractor(DistanceMethod):

    # name: str, agg_method: str, per_class: bool, per_stride: bool, cluster_method: str, cluster_optimization_metric: str
    def __init__(self,  **kwargs):
        name = 'ActivationsExtractor'
        per_class = True
        per_stride = True
        cluster_method = 'one'
        super().__init__(name, per_class, per_stride, cluster_method, **kwargs)

    def compute_distance(self, cluster: np.array, activations: np.array) -> List[float]:
        raise NotImplementedError("Not implemented yet")
    
    def activations_transformation(self, activations: np.array, **kwargs) -> np.array:
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

            # Convert imgs to float
            imgs = imgs.float() / 255.0

            ### Process the images to get the results (bboxes and clasification) and the extra info for OOD detection ###
            results = model.predict(imgs, save=False, verbose=False, device=device)

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
        super().__init__(name, per_class, per_stride, cluster_method, **kwargs)

    def compute_distance(self, cluster: np.array, activations: np.array) -> List[float]:
        raise NotImplementedError("Not implemented yet")
    
    def activations_transformation(self, activations: np.array, **kwargs) -> np.array:
        """
        Transform the activations to the shape needed to compute the distance.
        By default, it flattens the activations leaving the batch dimension as the first dimension.
        """
        raise NotImplementedError("Not implemented yet")
        return np.mean(activations, axis=(2,3))  # Already reshapes to [N, features]
    
    def iterate_data_to_extract_ind_activations_and_create_its_annotations(self, data_loader: InfiniteDataLoader, model: YOLO, device: str, split: str):
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

            # Convert imgs to float
            imgs = imgs.float() / 255.0

            ### Process the images to get the results (bboxes and clasification) and the extra info for OOD detection ###
            results = model.predict(imgs, save=False, verbose=False, device=device)

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
    

### FUSION Method ###

class FusionMethod(OODMethod):

    method1: Union[LogitsMethod, DistanceMethod]
    method2: Union[LogitsMethod, DistanceMethod]
    fusion_strategy: str

    def __init__(self, method1: Union[LogitsMethod, DistanceMethod], method2: Union[LogitsMethod, DistanceMethod], fusion_strategy: str, fusion_method_name: str,  cluster_method: str, **kwargs):

        self.method1 = method1
        self.method2 = method2
        self.fusion_strategy = fusion_strategy
        if method1.is_distance_method or method2.is_distance_method:
            is_distance_method = True
        else:
            is_distance_method = False
        super().__init__(name=fusion_method_name, per_class=True, per_stride=True, is_distance_method=is_distance_method, which_internal_activations="none", **kwargs)

        if is_distance_method:
            self.cluster_method = cluster_method
        else:
            self.cluster_method = 'None'

        # Define as properties the clusters and thresholds
        self._clusters = None
        self._thresholds = None
    
    # Clusters
    @property
    def clusters(self):
        if self.method1.is_distance_method and self.method2.is_distance_method:
            return self.method1.clusters, self.method2.clusters
        elif self.method1.is_distance_method:
            return self.method1.clusters
        elif self.method2.is_distance_method:
            return self.method2.clusters
        else:
            raise ValueError("This should not be called if none of the methods is a distance method")
    
    @clusters.setter
    def clusters(self, clusters):
        if self.method1.is_distance_method and self.method2.is_distance_method:
            self.method1.clusters = clusters[0]
            self.method2.clusters = clusters[1]
        elif self.method1.is_distance_method:
            self.method1.clusters = clusters
        elif self.method2.is_distance_method:
            self.method2.clusters = clusters
        else:
            raise ValueError("At least one of the methods must be a distance method to set the clusters")

    # Thresholds
    @property
    def thresholds(self):
        return self.method1.thresholds, self.method2.thresholds
    
    @thresholds.setter
    def thresholds(self, thresholds):
        if thresholds is not None:
            if len(thresholds) == 2:
                self.method1.thresholds = thresholds[0]
                self.method2.thresholds = thresholds[1]
            else:
                raise ValueError("The thresholds must be a tuple with two elements, one for the logits and one for the distance")
        else:
            print("The thresholds must be a tuple with two elements, one for the logits and one for the distance")
            self.method1.thresholds = None
            self.method2.thresholds = None


    def extract_internal_activations(self, results: Results, all_activations: Union[List[float], List[List[np.ndarray]]], targets: Dict[str, Tensor]):
        """
        Function to be overriden by each method to extract the internal activations of the model. In the logits
        methods, it will be the logits, and in the ftmaps methods, it will be the ftmaps.
        The extracted activations will be stored in the list all_activations
        """
        pass

    def format_internal_activations(self, all_activations: Union[List[float], List[List[np.ndarray]]]):
        """
        Function to be overriden by each method to format the internal activations of the model.
        The extracted activations will be stored in the list all_activations
        """
        pass

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
    
    def compute_scores(self, activations, *args, **kwargs) -> np.ndarray:
        """
        Function to compute the scores of the activations. Either for one box or multiple boxes.
        The function should be overriden by each method and should try to vectorize the computation as much as possible.
        """
        pass

    def activations_transformation(self, activations: np.array, **kwargs) -> np.array:
        raise NotImplementedError("This method is not going to be called directly")

    def compute_distance(self, centroids: np.array, features: np.array) -> np.array:
        raise NotImplementedError("This method is not going to be called directly")

    def iterate_data_to_extract_ind_activations(self, data_loader, model: YOLO, device: str, logger: Logger):
        
        configure_extra_output_of_the_model(model, which_internal_activations=self.method1.which_internal_activations)
        all_activations1 = self.method1.iterate_data_to_extract_ind_activations(data_loader, model, device, logger)
        configure_extra_output_of_the_model(model, which_internal_activations=self.method2.which_internal_activations)
        all_activations2 = self.method2.iterate_data_to_extract_ind_activations(data_loader, model, device, logger)

        return all_activations1, all_activations2

    def generate_thresholds(self, ind_scores: list, tpr: float, logger: Logger) -> Union[List[float], List[List[float]]]:
        
        activations1, activations2 = ind_scores
        thrs1 = self.method1.generate_thresholds(activations1, tpr, logger)
        thrs2 = self.method2.generate_thresholds(activations2, tpr, logger)

        return thrs1, thrs2

    def generate_clusters(self, ind_tensors: Union[List[np.ndarray], List[List[np.ndarray]]], logger: Logger) -> Union[List[np.ndarray], List[List[np.ndarray]]]:
        if self.method1.is_distance_method and self.method2.is_distance_method:
            clusters1 = self.method1.generate_clusters(ind_tensors[0], logger)
            clusters2 = self.method2.generate_clusters(ind_tensors[1], logger)
            return [clusters1, clusters2]
        elif self.method1.is_distance_method:
            return self.method1.generate_clusters(ind_tensors[0], logger)
        elif self.method2.is_distance_method:
            return self.method2.generate_clusters(ind_tensors[1], logger)
        else:
            raise ValueError("Both methods must be distance methods to generate the clusters")
    
    def compute_scores_from_activations(self, activations: Union[List[np.ndarray], List[List[np.ndarray]]], logger: Logger) -> Tuple[List[List[float]], List[List[float]]]:
        
        activations1, activations2 = activations
        scores1 = self.method1.compute_scores_from_activations(activations1, logger)
        scores2 = self.method2.compute_scores_from_activations(activations2, logger)

        return scores1, scores2

    def fuse_ood_decisions(self, ood_decision1: List[List[int]], ood_decision2: List[List[int]]) -> List[List[int]]:
        # 1 is InD, 0 is OoD
        ood_decision = []
        if self.fusion_strategy == "and":
            # AND strategy: Only if both methods say it is OoD (decision = 0), then it is OoD
            #   Put it other way: If one of methods say that the bbox is InD (decision = 1), then it is InD
            for idx_img in range(len(ood_decision1)):
                ood_decision.append(
                    [max(ood_decision1[idx_img][idx_bbox], ood_decision2[idx_img][idx_bbox]) for idx_bbox in range(len(ood_decision1[idx_img]))]
                )

        elif self.fusion_strategy == "or":
            # OR strategy: If one of the methods says that the bbox is OoD (decision = 0), then it is OoD
            for idx_img in range(len(ood_decision1)):
                ood_decision.append(
                    [min(ood_decision1[idx_img][idx_bbox], ood_decision2[idx_img][idx_bbox]) for idx_bbox in range(len(ood_decision1[idx_img]))]
                )
        
        elif self.fusion_strategy == "score":
            # SCORE strategy: The score is the sum of the scores of the two methods. If the score is greater than 0, it is InD
            for idx_img in range(len(ood_decision1)):
                ood_score_one_img = [ood_decision1[idx_img][idx_bbox] + ood_decision2[idx_img][idx_bbox] for idx_bbox in range(len(ood_decision1[idx_img]))]
                ood_decision.append(
                    [1 if score > 0 else 0 for score in ood_score_one_img]                    
                )
            
        else:
            raise NotImplementedError("Not implemented yet")
        
        # Assert that the number of bboxes is the same after the decision
        for idx_img in range(len(ood_decision)):
            assert len(ood_decision[idx_img]) == len(ood_decision1[idx_img]), "The number of bboxes is different"
            assert len(ood_decision[idx_img]) == len(ood_decision2[idx_img]), "The number of bboxes is different"

        return ood_decision

    def iterate_data_to_compute_metrics(self, model: YOLO, device: str, dataloader: InfiniteDataLoader, logger: Logger, known_classes: List[int]) -> Dict[str, float]:
        
        logger.warning(f"Using a confidence threshold of {self.min_conf_threshold_test} for tests")
        number_of_images_processed = 0
        number_of_batches = len(dataloader)
        all_preds = []
        all_targets = []
        assert hasattr(dataloader.dataset, "number_of_classes"), "The dataset does not have the attribute number_of_classes to know the number of classes known in the dataset"
        class_names = list(dataloader.dataset.data['names'].values())[:dataloader.dataset.number_of_classes]
        class_names.append('unknown')
        known_classes_tensor = torch.tensor(known_classes, dtype=torch.float32)

        number_of_images_saved = 0
        count_of_images = 0
        for idx_of_batch, data in enumerate(dataloader):
            count_of_images += len(data['im_file'])

            if idx_of_batch % 50 == 0 or idx_of_batch == number_of_batches - 1:
                logger.info(f"{(idx_of_batch/number_of_batches) * 100:02.1f}%: Procesing batch {idx_of_batch+1} of {number_of_batches}") 

            ### Preparar imagenes y targets ###
            imgs, targets = self.prepare_data_for_model(data, device)

            # Convert imgs to float
            imgs = imgs.float() / 255

            ###
            # Method 1
            ###
            configure_extra_output_of_the_model(model, ood_method=self.method1)
            ### Procesar imagenes en el modelo para obtener logits y las cajas ###
            results_1 = model.predict(imgs, save=False, verbose=False, conf=self.min_conf_threshold_test, device=device)
            ### Comprobar si las cajas predichas son OoD ###
            if self.fusion_strategy == 'score':
                ood_decision1 = self.method1.compute_INDness_scores_on_results(results_1, logger)
            else:
                ood_decision1 = self.method1.compute_ood_decision_on_results(results_1, logger)

            ###
            # Method 2
            ###
            configure_extra_output_of_the_model(model, ood_method=self.method2)
            ### Procesar imagenes en el modelo para obtener las caracteristicas y las cajas ###
            results_2 = model.predict(imgs, save=False, verbose=False, conf=self.min_conf_threshold_test, device=device)
            ### Comprobar si las cajas predichas son OoD ###
            if self.fusion_strategy == 'score':
                ood_decision2 = self.method2.compute_INDness_scores_on_results(results_2, logger)
            else:
                ood_decision2 = self.method2.compute_ood_decision_on_results(results_2, logger)

            # Assert deeply that results are the same and assign the results as one of them (either logits or distance)
            for idx_r in range(len(results_1)):
                assert torch.allclose(results_1[idx_r].boxes.xyxy, results_2[idx_r].boxes.xyxy), f"Results are not the same for image {idx_r}"
                assert torch.allclose(results_1[idx_r].boxes.cls, results_2[idx_r].boxes.cls), f"Results are not the same for image {idx_r}"
                assert torch.allclose(results_1[idx_r].boxes.conf, results_2[idx_r].boxes.conf), f"Results are not the same for image {idx_r}"
            results = results_1

            ### Fuse the results of the logits and distance methods ###
            ood_decision = self.fuse_ood_decisions(ood_decision1, ood_decision2)

            ### Añadir posibles cajas desconocidas a las predicciones ###
            if self.enhanced_unk_localization:
                if CUSTOM_HYP.unk.RANK_BOXES:
                        possible_unk_bboxes, ood_decision_on_unknown, distances_per_image = self.compute_extra_possible_unkwnown_bboxes_and_decision(
                            results, data, ood_decision_of_results=ood_decision, folder_path=None,
                            origin_of_idx=idx_of_batch*dataloader.batch_size
                        )
                else:
                    distances_per_image = None
                    possible_unk_bboxes, ood_decision_on_unknown = self.compute_extra_possible_unkwnown_bboxes_and_decision(
                        results, data, ood_decision_of_results=ood_decision, folder_path=None,
                        origin_of_idx=idx_of_batch*dataloader.batch_size
                    )

            # Cada prediccion va a ser un diccionario con las siguientes claves:
            #   'img_idx': int -> Indice de la imagen
            #   'img_name': str -> Nombre del archivo de la imagen
            #   'bboxes': List[Tensor] -> Lista de tensores con las cajas predichas
            #   'cls': List[Tensor] -> Lista de tensores con las clases predichas
            #   'conf': List[Tensor] -> Lista de tensores con las confianzas de las predicciones (en yolov8 es cls)
            #   'ood_decision': List[int] -> Lista de enteros con la decision de si la caja es OoD o no
            for img_idx, res in enumerate(results):
                #for idx_bbox in range(len(res.boxes.cls)):
                # if self.enhanced_unk_localization:
                #     pass
                # else:
                # Parse the ood elements as the unknown class (80)
                ood_decision_one_image = torch.tensor(ood_decision[img_idx], dtype=torch.float32)
                unknown_mask = ood_decision_one_image == 0
                bboxes_coords = res.boxes.xyxy.cpu()
                bboxes_cls = torch.where(unknown_mask, torch.tensor(80, dtype=torch.float32), res.boxes.cls.cpu())   
                # Make all the preds to be the class 80 to known the max recall possible
                #bboxes_cls = torch.tensor(80, dtype=torch.float32).repeat(len(res.boxes.cls))
                bboxes_conf = res.boxes.conf.cpu()
                # TODO: La logica de como ignorar ciertos unknowns la tenemos que idear, ya que por el momento no tiene 
                #   sentido que las propuestas no sean unknowns
                if self.enhanced_unk_localization:
                    # Add the possible unknown boxes to the predictions
                    bboxes_coords = torch.cat([bboxes_coords, possible_unk_bboxes[img_idx]], dim=0)
                    one_image_ood_decision_on_possible_unk = torch.tensor(ood_decision_on_unknown[img_idx], dtype=torch.float32)
                    assert one_image_ood_decision_on_possible_unk.sum().item() == 0.0, "Uno de los posible unknowns es considerado como known, pero como no tenemos esa logica implementada es ERROR"
                    # TODO: Por el momento simplemente lo que hago es hacer un tensor con todo clase 80 (unk). Luego tendre que ver como gestiono el hacer que acaben siendo una clase
                    cls_unk_prop = torch.tensor(80, dtype=torch.float32).repeat(len(possible_unk_bboxes[img_idx]))
                    bboxes_cls = torch.cat([bboxes_cls, cls_unk_prop], dim=0)
                    conf_unk_prop = torch.ones(len(possible_unk_bboxes[img_idx])) * 0.150001  # TODO: Pongo 0.150001 de confianza para las propuestas de unknown
                    bboxes_conf = torch.cat([bboxes_conf, conf_unk_prop], dim=0)
                all_preds.append({
                    'img_idx': number_of_images_processed + img_idx,
                    'img_name': Path(data['im_file'][img_idx]).stem,
                    'bboxes': bboxes_coords,
                    'cls': bboxes_cls,
                    'conf': bboxes_conf,
                    #'ood_decision': torch.tensor(ood_decision[img_idx], dtype=torch.float32)
                })
                    
                # Transform the classes to index 80 if they are not in the known classes
                known_mask = torch.isin(targets['cls'][img_idx], known_classes_tensor)
                transformed_target_cls = torch.where(known_mask, targets['cls'][img_idx], torch.tensor(80, dtype=torch.float32))
                all_targets.append({
                    'img_idx': number_of_images_processed + img_idx,
                    'img_name': Path(data['im_file'][img_idx]).stem,
                    'bboxes': targets['bboxes'][img_idx],
                    'cls': transformed_target_cls
                })

            ### Acumular predicciones y targets ###
            number_of_images_processed += len(imgs)
            number_of_images_saved += len(data['im_file'])

        #########
        # Loop finished
        #########

        # All predictions collected, now compute metrics
        results_dict = compute_metrics(all_preds, all_targets, class_names, known_classes, logger)

        # Count the number of non-unknown instances and the number of unknown instances
        number_of_known_boxes = 0 
        number_of_unknown_boxes = 0
        for _target in all_targets:
            number_of_known_boxes += torch.sum(_target['cls'] != 80).item()
            number_of_unknown_boxes += torch.sum(_target['cls'] == 80).item()
        logger.info(f"Number of target known boxes: {number_of_known_boxes}")
        logger.info(f"Number of target unknown boxes: {number_of_unknown_boxes}")

        return results_dict


### FUSION Method ###

class TripleFusionMethod(OODMethod):

    method1: Union[LogitsMethod, DistanceMethod]
    method2: Union[LogitsMethod, DistanceMethod]
    method3: Union[LogitsMethod, DistanceMethod]
    fusion_strategy: str

    def __init__(self, method1: Union[LogitsMethod, DistanceMethod], method2: Union[LogitsMethod, DistanceMethod], method3: Union[LogitsMethod, DistanceMethod], cluster_method: str, **kwargs):

        name = f'fusion-{method1.name}-{method2.name}_{method3.name}'
        self.method1 = method1
        self.method2 = method2
        self.method3 = method3
        self.fusion_strategy = 'majority_voting'
        if method1.is_distance_method or method2.is_distance_method or method3.is_distance_method:
            is_distance_method = True
        else:
            is_distance_method = False
        super().__init__(name=name, per_class=True, per_stride=True, is_distance_method=is_distance_method, which_internal_activations="none", **kwargs)

        if is_distance_method:
            self.cluster_method = cluster_method
        else:
            self.cluster_method = 'None'

        # Define as properties the clusters and thresholds
        self._clusters = None
        self._thresholds = None
    
    # Clusters
    @property
    def clusters(self):
        if self.method1.is_distance_method and self.method2.is_distance_method and self.method3.is_distance_method:
            return self.method1.clusters, self.method2.clusters, self.method3.clusters
        elif self.method1.is_distance_method and self.method2.is_distance_method:
            return self.method1.clusters, self.method2.clusters
        elif self.method1.is_distance_method and self.method3.is_distance_method:
            return self.method1.clusters, self.method3.clusters
        elif self.method2.is_distance_method and self.method3.is_distance_method:
            return self.method2.clusters, self.method3.clusters
        elif self.method1.is_distance_method:
            return self.method1.clusters
        elif self.method2.is_distance_method:
            return self.method2.clusters
        elif self.method3.is_distance_method:
            return self.method3.clusters
        else:
            raise ValueError("This should not be called if none of the methods is a distance method")
    
    @clusters.setter
    def clusters(self, clusters):
        if self.method1.is_distance_method and self.method2.is_distance_method and self.method3.is_distance_method:
            self.method1.clusters = clusters[0]
            self.method2.clusters = clusters[1]
            self.method3.clusters = clusters[2]
        elif self.method1.is_distance_method and self.method2.is_distance_method:
            self.method1.clusters = clusters[0]
            self.method2.clusters = clusters[1]
        elif self.method1.is_distance_method and self.method3.is_distance_method:
            self.method1.clusters = clusters[0]
            self.method3.clusters = clusters[1]
        elif self.method2.is_distance_method and self.method3.is_distance_method:
            self.method2.clusters = clusters[0]
            self.method3.clusters = clusters[1]
        elif self.method1.is_distance_method:
            self.method1.clusters = clusters
        elif self.method2.is_distance_method:
            self.method2.clusters = clusters
        elif self.method3.is_distance_method:
            self.method3.clusters = clusters
        else:
            raise ValueError("At least one of the methods must be a distance method to set the clusters")

    # Thresholds
    @property
    def thresholds(self):
        return self.method1.thresholds, self.method2.thresholds, self.method3.thresholds
    
    @thresholds.setter
    def thresholds(self, thresholds):
        if thresholds is not None:
            if len(thresholds) == 3:
                self.method1.thresholds = thresholds[0]
                self.method2.thresholds = thresholds[1]
                self.method3.thresholds = thresholds[2]
            else:
                raise ValueError("The thresholds must be a tuple with two elements, one for the logits and one for the distance")
        else:
            print("The thresholds must be a tuple with two elements, one for the logits and one for the distance")
            self.method1.thresholds = None
            self.method2.thresholds = None
            self.method3.thresholds = None


    def extract_internal_activations(self, results: Results, all_activations: Union[List[float], List[List[np.ndarray]]], targets: Dict[str, Tensor]):
        """
        Function to be overriden by each method to extract the internal activations of the model. In the logits
        methods, it will be the logits, and in the ftmaps methods, it will be the ftmaps.
        The extracted activations will be stored in the list all_activations
        """
        pass

    def format_internal_activations(self, all_activations: Union[List[float], List[List[np.ndarray]]]):
        """
        Function to be overriden by each method to format the internal activations of the model.
        The extracted activations will be stored in the list all_activations
        """
        pass

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
    
    def compute_scores(self, activations, *args, **kwargs) -> np.ndarray:
        """
        Function to compute the scores of the activations. Either for one box or multiple boxes.
        The function should be overriden by each method and should try to vectorize the computation as much as possible.
        """
        pass

    def activations_transformation(self, activations: np.array, **kwargs) -> np.array:
        raise NotImplementedError("This method is not going to be called directly")

    def compute_distance(self, centroids: np.array, features: np.array) -> np.array:
        raise NotImplementedError("This method is not going to be called directly")

    def iterate_data_to_extract_ind_activations(self, data_loader, model: YOLO, device: str, logger: Logger):
        
        configure_extra_output_of_the_model(model, which_internal_activations=self.method1.which_internal_activations)
        all_activations1 = self.method1.iterate_data_to_extract_ind_activations(data_loader, model, device, logger)
        configure_extra_output_of_the_model(model, which_internal_activations=self.method2.which_internal_activations)
        all_activations2 = self.method2.iterate_data_to_extract_ind_activations(data_loader, model, device, logger)
        configure_extra_output_of_the_model(model, which_internal_activations=self.method3.which_internal_activations)
        all_activations3 = self.method3.iterate_data_to_extract_ind_activations(data_loader, model, device, logger)

        return all_activations1, all_activations2, all_activations3

    def generate_thresholds(self, ind_scores: list, tpr: float, logger: Logger) -> Union[List[float], List[List[float]]]:
        
        activations1, activations2, activations3 = ind_scores
        thrs1 = self.method1.generate_thresholds(activations1, tpr, logger)
        thrs2 = self.method2.generate_thresholds(activations2, tpr, logger)
        thrs3 = self.method3.generate_thresholds(activations3, tpr, logger)

        return thrs1, thrs2, thrs3

    def generate_clusters(self, ind_tensors: Union[List[np.ndarray], List[List[np.ndarray]]], logger: Logger) -> Union[List[np.ndarray], List[List[np.ndarray]]]:
        if self.method1.is_distance_method and self.method2.is_distance_method and self.method3.is_distance_method:
            clusters1 = self.method1.generate_clusters(ind_tensors[0], logger)
            clusters2 = self.method2.generate_clusters(ind_tensors[1], logger)
            clusters3 = self.method3.generate_clusters(ind_tensors[2], logger)
            return [clusters1, clusters2, clusters3]
        if self.method1.is_distance_method and self.method2.is_distance_method:
            clusters1 = self.method1.generate_clusters(ind_tensors[0], logger)
            clusters2 = self.method2.generate_clusters(ind_tensors[1], logger)
            return [clusters1, clusters2]
        if self.method1.is_distance_method and self.method3.is_distance_method:
            clusters1 = self.method1.generate_clusters(ind_tensors[0], logger)
            clusters3 = self.method3.generate_clusters(ind_tensors[1], logger)
            return [clusters1, clusters3]
        if self.method2.is_distance_method and self.method3.is_distance_method:
            clusters2 = self.method2.generate_clusters(ind_tensors[0], logger)
            clusters3 = self.method3.generate_clusters(ind_tensors[1], logger)
            return [clusters2, clusters3]
        elif self.method1.is_distance_method:
            return self.method1.generate_clusters(ind_tensors[0], logger)
        elif self.method2.is_distance_method:
            return self.method2.generate_clusters(ind_tensors[1], logger)
        elif self.method3.is_distance_method:
            return self.method3.generate_clusters(ind_tensors[2], logger)
        else:
            raise ValueError("Both methods must be distance methods to generate the clusters")
    
    def compute_scores_from_activations(self, activations: Union[List[np.ndarray], List[List[np.ndarray]]], logger: Logger) -> Tuple[List[List[float]], List[List[float]]]:
        
        activations1, activations2, activations3 = activations
        scores1 = self.method1.compute_scores_from_activations(activations1, logger)
        scores2 = self.method2.compute_scores_from_activations(activations2, logger)
        scores3 = self.method3.compute_scores_from_activations(activations3, logger)

        return scores1, scores2, scores3

    def fuse_ood_decisions(self, ood_decision1: List[List[int]], ood_decision2: List[List[int]], ood_decision3: List[List[int]]) -> List[List[int]]:
        # 1 is InD, 0 is OoD
        ood_decision = []
        if self.fusion_strategy == 'majority_voting':
            # Majority voting: If two of the three methods say that the bbox is InD (decision = 1), then it is InD
            for idx_img in range(len(ood_decision1)):
                ood_decision.append(
                    [1 if (ood_decision1[idx_img][idx_bbox] + ood_decision2[idx_img][idx_bbox] + ood_decision3[idx_img][idx_bbox]) >= 2 else 0 for idx_bbox in range(len(ood_decision1[idx_img]))]
                )
            
        else:
            raise ValueError("Only valid majority_voting fusion strategy")
        
        # Assert that the number of bboxes is the same after the decision
        for idx_img in range(len(ood_decision)):
            assert len(ood_decision[idx_img]) == len(ood_decision1[idx_img]), "The number of bboxes is different"
            assert len(ood_decision[idx_img]) == len(ood_decision2[idx_img]), "The number of bboxes is different"
            assert len(ood_decision[idx_img]) == len(ood_decision3[idx_img]), "The number of bboxes is different"

        return ood_decision

    def iterate_data_to_compute_metrics(self, model: YOLO, device: str, dataloader: InfiniteDataLoader, logger: Logger, known_classes: List[int]) -> Dict[str, float]:
        
        logger.warning(f"Using a confidence threshold of {self.min_conf_threshold_test} for tests")
        number_of_images_processed = 0
        number_of_batches = len(dataloader)
        all_preds = []
        all_targets = []
        assert hasattr(dataloader.dataset, "number_of_classes"), "The dataset does not have the attribute number_of_classes to know the number of classes known in the dataset"
        class_names = list(dataloader.dataset.data['names'].values())[:dataloader.dataset.number_of_classes]
        class_names.append('unknown')
        known_classes_tensor = torch.tensor(known_classes, dtype=torch.float32)
        number_of_images_saved = 0
        count_of_images = 0
        for idx_of_batch, data in enumerate(dataloader):
            count_of_images += len(data['im_file'])

            if idx_of_batch % 50 == 0 or idx_of_batch == number_of_batches - 1:
                logger.info(f"{(idx_of_batch/number_of_batches) * 100:02.1f}%: Procesing batch {idx_of_batch+1} of {number_of_batches}") 

            ### Preparar imagenes y targets ###
            imgs, targets = self.prepare_data_for_model(data, device)

            # Convert imgs to float
            imgs = imgs.float() / 255

            ###
            # Method 1
            ###
            configure_extra_output_of_the_model(model, ood_method=self.method1)
            ### Procesar imagenes en el modelo para obtener logits y las cajas ###
            results_1 = model.predict(imgs, save=False, verbose=False, conf=self.min_conf_threshold_test, device=device)
            ### Comprobar si las cajas predichas son OoD ###
            ood_decision1 = self.method1.compute_ood_decision_on_results(results_1, logger)

            ###
            # Method 2
            ###
            configure_extra_output_of_the_model(model, ood_method=self.method2)
            ### Procesar imagenes en el modelo para obtener las caracteristicas y las cajas ###
            results_2 = model.predict(imgs, save=False, verbose=False, conf=self.min_conf_threshold_test, device=device)
            ### Comprobar si las cajas predichas son OoD ###
            ood_decision2 = self.method2.compute_ood_decision_on_results(results_2, logger)

            ###
            # Method 3
            ###
            configure_extra_output_of_the_model(model, ood_method=self.method3)
            ### Procesar imagenes en el modelo para obtener las caracteristicas y las cajas ###
            results_3 = model.predict(imgs, save=False, verbose=False, conf=self.min_conf_threshold_test, device=device)
            ### Comprobar si las cajas predichas son OoD ###
            ood_decision3 = self.method3.compute_ood_decision_on_results(results_3, logger)

            # Assert deeply that results are the same and assign the results as one of them (either logits or distance)
            for idx_r in range(len(results_1)):
                assert torch.allclose(results_1[idx_r].boxes.xyxy, results_2[idx_r].boxes.xyxy), f"Results are not the same for image {idx_r}"
                assert torch.allclose(results_1[idx_r].boxes.cls, results_2[idx_r].boxes.cls), f"Results are not the same for image {idx_r}"
                assert torch.allclose(results_1[idx_r].boxes.conf, results_2[idx_r].boxes.conf), f"Results are not the same for image {idx_r}"
                assert torch.allclose(results_1[idx_r].boxes.xyxy, results_3[idx_r].boxes.xyxy), f"Results are not the same for image {idx_r}"
                assert torch.allclose(results_1[idx_r].boxes.cls, results_3[idx_r].boxes.cls), f"Results are not the same for image {idx_r}"
                assert torch.allclose(results_1[idx_r].boxes.conf, results_3[idx_r].boxes.conf), f"Results are not the same for image {idx_r}"
            results = results_1

            ### Fuse the results of the logits and distance methods ###
            ood_decision = self.fuse_ood_decisions(ood_decision1, ood_decision2, ood_decision3)

            ### Añadir posibles cajas desconocidas a las predicciones ###
            if self.enhanced_unk_localization:
                if CUSTOM_HYP.unk.RANK_BOXES:
                        possible_unk_bboxes, ood_decision_on_unknown, distances_per_image = self.compute_extra_possible_unkwnown_bboxes_and_decision(
                            results, data, ood_decision_of_results=ood_decision, folder_path=None, origin_of_idx=idx_of_batch*dataloader.batch_size
                        )
                else:
                    distances_per_image = None
                    possible_unk_bboxes, ood_decision_on_unknown = self.compute_extra_possible_unkwnown_bboxes_and_decision(
                        results, data, ood_decision_of_results=ood_decision, folder_path=None, origin_of_idx=idx_of_batch*dataloader.batch_size
                    )

            # Cada prediccion va a ser un diccionario con las siguientes claves:
            #   'img_idx': int -> Indice de la imagen
            #   'img_name': str -> Nombre del archivo de la imagen
            #   'bboxes': List[Tensor] -> Lista de tensores con las cajas predichas
            #   'cls': List[Tensor] -> Lista de tensores con las clases predichas
            #   'conf': List[Tensor] -> Lista de tensores con las confianzas de las predicciones (en yolov8 es cls)
            #   'ood_decision': List[int] -> Lista de enteros con la decision de si la caja es OoD o no
            for img_idx, res in enumerate(results):
                #for idx_bbox in range(len(res.boxes.cls)):
                # if self.enhanced_unk_localization:
                #     pass
                # else:
                # Parse the ood elements as the unknown class (80)
                ood_decision_one_image = torch.tensor(ood_decision[img_idx], dtype=torch.float32)
                unknown_mask = ood_decision_one_image == 0
                bboxes_coords = res.boxes.xyxy.cpu()
                bboxes_cls = torch.where(unknown_mask, torch.tensor(80, dtype=torch.float32), res.boxes.cls.cpu())   
                # Make all the preds to be the class 80 to known the max recall possible
                #bboxes_cls = torch.tensor(80, dtype=torch.float32).repeat(len(res.boxes.cls))
                bboxes_conf = res.boxes.conf.cpu()
                # TODO: La logica de como ignorar ciertos unknowns la tenemos que idear, ya que por el momento no tiene 
                #   sentido que las propuestas no sean unknowns
                if self.enhanced_unk_localization:
                    # Add the possible unknown boxes to the predictions
                    bboxes_coords = torch.cat([bboxes_coords, possible_unk_bboxes[img_idx]], dim=0)
                    one_image_ood_decision_on_possible_unk = torch.tensor(ood_decision_on_unknown[img_idx], dtype=torch.float32)
                    assert one_image_ood_decision_on_possible_unk.sum().item() == 0.0, "Uno de los posible unknowns es considerado como known, pero como no tenemos esa logica implementada es ERROR"
                    # TODO: Por el momento simplemente lo que hago es hacer un tensor con todo clase 80 (unk). Luego tendre que ver como gestiono el hacer que acaben siendo una clase
                    cls_unk_prop = torch.tensor(80, dtype=torch.float32).repeat(len(possible_unk_bboxes[img_idx]))
                    bboxes_cls = torch.cat([bboxes_cls, cls_unk_prop], dim=0)
                    conf_unk_prop = torch.ones(len(possible_unk_bboxes[img_idx])) * 0.150001  # TODO: Pongo 0.150001 de confianza para las propuestas de unknown
                    bboxes_conf = torch.cat([bboxes_conf, conf_unk_prop], dim=0)
                all_preds.append({
                    'img_idx': number_of_images_processed + img_idx,
                    'img_name': Path(data['im_file'][img_idx]).stem,
                    'bboxes': bboxes_coords,
                    'cls': bboxes_cls,
                    'conf': bboxes_conf,
                    #'ood_decision': torch.tensor(ood_decision[img_idx], dtype=torch.float32)
                })
                    
                # Transform the classes to index 80 if they are not in the known classes
                known_mask = torch.isin(targets['cls'][img_idx], known_classes_tensor)
                transformed_target_cls = torch.where(known_mask, targets['cls'][img_idx], torch.tensor(80, dtype=torch.float32))
                all_targets.append({
                    'img_idx': number_of_images_processed + img_idx,
                    'img_name': Path(data['im_file'][img_idx]).stem,
                    'bboxes': targets['bboxes'][img_idx],
                    'cls': transformed_target_cls
                })

            ### Acumular predicciones y targets ###
            number_of_images_processed += len(imgs)
            number_of_images_saved += len(data['im_file'])

        #########
        # Loop finished
        #########

        # All predictions collected, now compute metrics
        results_dict = compute_metrics(all_preds, all_targets, class_names, known_classes, logger)

        # Count the number of non-unknown instances and the number of unknown instances
        number_of_known_boxes = 0 
        number_of_unknown_boxes = 0
        for _target in all_targets:
            number_of_known_boxes += torch.sum(_target['cls'] != 80).item()
            number_of_unknown_boxes += torch.sum(_target['cls'] == 80).item()
        logger.info(f"Number of target known boxes: {number_of_known_boxes}")
        logger.info(f"Number of target unknown boxes: {number_of_unknown_boxes}")

        return results_dict


# class CosineIvisPerStrideOnly(_IvisMethodPairwiseDistance):
#     def __init__(self, name: str, metric: str, **kwargs):
#         super().__init__(**kwargs)            

#     # Only modify the method to compute the distance, as we want to compute the distance against ALL classes in same stride
#     def _compute_ood_decision_for_one_result_from_roi_aligned_feature_maps(
#             self, idx_img: int, one_img_bboxes_cls_idx: Tensor, roi_aligned_ftmaps_one_img_per_stride, ood_decision: List, logger: Logger
#     ):
#         """
#         Compute the OOD decision for one image using the in-distribution activations (usually feature maps).
#         Pipeline:
#             1. Loop over the strides. Each stride is a list of bboxes and their feature maps
#             2. Compute the distance between the prediction and the cluster of the predicted class
#             3. Compare the distance with the threshold
#         """
#         # Loop each stride of the image. Select the first element of the list as we are processing one image only
#         for stride_idx, (bbox_idx_in_one_stride, ftmaps) in enumerate(roi_aligned_ftmaps_one_img_per_stride):
            
#             # Only enter if there are any predictions in this stride
#             if len(bbox_idx_in_one_stride) > 0:
#                 # Each ftmap is from a bbox prediction
#                 for idx, ftmap in enumerate(ftmaps):
#                     bbox_idx = idx
#                     cls_idx = int(one_img_bboxes_cls_idx[bbox_idx])
#                     ftmap = ftmap.cpu().unsqueeze(0).numpy()  # To obtain a tensor of shape [1, C, H, W]
#                     # ftmap = ftmap.cpu().flatten().unsqueeze(0).numpy()
#                     # [None, :] is to do the same as unsqueeze(0) but with numpy
#                     # Check if there is a cluster for the class and stride
#                     if len(self.clusters[cls_idx][stride_idx]) == 0:
#                         logger.warning(
#                             f'Image {idx_img}, bbox {bbox_idx} is viewed as an OOD.' \
#                             f'It cannot be compared as there is no cluster for class {cls_idx} and stride {stride_idx}'
#                         )
#                         distance = 1000
#                     else:
#                         # Collect all the clusters for this stride
#                         cluster_of_stride = []
#                         for cls_idx in range(len(self.clusters)):
#                             if len(self.clusters[cls_idx][stride_idx]) > 0:
#                                 cluster_of_stride.append(self.clusters[cls_idx][stride_idx])
#                         cluster_of_stride = np.concatenate(cluster_of_stride, axis=0)
#                         distance = self.compute_distance(
#                             cluster_of_stride,
#                             self.activations_transformation(ftmap, cls_idx=cls_idx, stride_idx=stride_idx)
#                         )[0]

#                     # Check if the distance is lower than the threshold
#                     if self.thresholds[cls_idx][stride_idx]:
#                         if distance < self.thresholds[cls_idx][stride_idx]:
#                             ood_decision[idx_img].append(1)  # InD
#                         else:
#                             ood_decision[idx_img].append(0)  # OOD
#                     else:
#                         # logger.warning(f'WARNING: Class {cls_idx:03}, Stride {stride_idx} -> No threshold!')
#                         ood_decision[idx_img].append(0)  # OOD   
        
#     def compute_distance(self, cluster: np.array, activations: np.array) -> List[float]:

#         distances = pairwise_distances(
#             cluster,
#             activations,
#             metric=self.metric
#             )

#         return distances.min(axis=0)


### Method to configure internals of the model ###
    
def configure_extra_output_of_the_model(model: YOLO, ood_method: Type[OODMethod]):
        # Modify the model's attributes to output the desired extra_item
        # 1. Select the layers to extract depending on the OOD method from ultralytics/nn/tasks.py
        model.model.model[-1].output_values_before_sigmoid = False  # By default, we do not want to output the values before the sigmoid
        if ood_method.which_internal_activations in FTMAPS_RELATED_OPTIONS:
            model.model.which_layers_to_extract = "convolutional_layers"
        elif ood_method.which_internal_activations in LOGITS_RELATED_OPTIONS:
            model.model.which_layers_to_extract = "logits"
            if ood_method.use_values_before_sigmoid:
                model.model.model[-1].output_values_before_sigmoid = True
        elif ood_method.which_internal_activations == "none":
            model.model.which_layers_to_extract = "none"
        else:
            raise ValueError(f"The option {ood_method.which_internal_activations} is not valid.")
        # 2. Select the extraction mode for the ultralytics/yolo/v8/detect/predict.py
        model.model.extraction_mode = ood_method.which_internal_activations  # This attribute is created in the DetectionModel class

        if "yolov10" in model.ckpt_path:
            model.model.model[23].validating = False
