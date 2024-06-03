from typing import List, Tuple, Callable, Type, Union, Dict, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
from logging import Logger
import json
import inspect
import os

import numpy as np
# import sklearn.metrics as sk
from numpy.core.multiarray import array as array
from sklearn.metrics import pairwise_distances
import torch
import torch.nn.functional as F
from torch import Tensor
import torchvision.ops as t_ops
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment

from ultralytics import YOLO
from ultralytics.yolo.data.build import InfiniteDataLoader
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.v8.detect.predict import extract_roi_aligned_features_from_correct_stride
from visualization_utils import plot_results
from datasets_utils.owod.owod_evaluation_protocol import compute_metrics
from unknown_localization_utils import extract_bboxes_from_saliency_map_and_thresholds
from constants import IND_INFO_CREATION_OPTIONS, AVAILABLE_CLUSTERING_METHODS, \
    AVAILABLE_CLUSTER_OPTIMIZATION_METRICS, INTERNAL_ACTIVATIONS_EXTRACTION_OPTIONS, \
    FTMAPS_RELATED_OPTIONS, LOGITS_RELATED_OPTIONS, STRIDES_RATIO, IMAGE_FORMAT
from custom_hyperparams import CUSTOM_HYP
from YOLOv8_Explainer import yolov8_heatmap
from cluster_utils import find_optimal_number_of_clusters_one_class_one_stride_and_return_labels


import datetime
NOW_ood_utils = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def limit_heatmaps_to_bounding_boxes(expl_heatmaps: List[Tensor], results: List[Results]) -> List[Tensor]:
    processed_heatmaps = []
    for _i, expl_heatmaps_one_image in enumerate(expl_heatmaps):
        if expl_heatmaps_one_image is None:
            processed_heatmap_one_image = np.zeros((80, 80), dtype=np.float32)
        else:
            processed_heatmap_one_image = np.zeros(expl_heatmaps_one_image.shape, dtype=np.float32)
            boxes = results[_i].boxes.xyxy.cpu().numpy()
            orig_h, orig_w = results[_i].orig_img.shape[2:]
            # Convert boxes to heatmap size
            boxes = boxes * np.array(
                # The array is the reduction that has to be made in xyxy boxes format. The values of it are correspoding to (x1, y1, x2, y2)
                [expl_heatmaps_one_image.shape[1] / orig_w, expl_heatmaps_one_image.shape[0] / orig_h,  expl_heatmaps_one_image.shape[1] / orig_w, expl_heatmaps_one_image.shape[0] / orig_h]
            )
            boxes = boxes.astype(int)
            # # Draw bounding boxes over the heatmap
            # import matplotlib.pyplot as plt
            # im = draw_bounding_boxes(
            #     (torch.tensor(expl_heatmaps_one_image).unsqueeze(0) * 255).to(torch.uint8),
            #     torch.tensor(boxes),
            #     width=2,
            #     font='FreeMonoBold',
            #     font_size=11,
            #     colors=['red']*len(boxes)
            # )
            # plt.imshow(im.permute(1,2,0))
            # plt.savefig('EEEEE.png')
            # plt.close()

            for x1, y1, x2, y2 in boxes:
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(expl_heatmaps_one_image.shape[1] - 1, x2), min(expl_heatmaps_one_image.shape[0] - 1, y2)
                processed_heatmap_one_image[y1:y2, x1:x2] = expl_heatmaps_one_image[y1:y2, x1:x2].clone()

        processed_heatmaps.append(torch.tensor(processed_heatmap_one_image, dtype=torch.float32))
        
    return processed_heatmaps
    

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
    enhanced_unk_localization: bool  # If True, the method will try to enhance the localization of the UNK objects by adding new boxes
    compute_saliency_map_one_stride: Callable  # Function to compute the saliency map of one stride
    compute_thresholds_out_of_saliency_map: Callable  # Function to compute the thresholds out of the saliency map
    #extract_bboxes_from_saliency_map_and_thresholds: Callable  # Function to extract the bboxes from the saliency map and the thresholds

    def __init__(self, name: str, is_distance_method: bool, per_class: bool, per_stride: bool, iou_threshold_for_matching: float,
                 min_conf_threshold_train: float, min_conf_threshold_test: float, which_internal_activations: str, enhanced_unk_localization: bool = False,
                 saliency_map_computation_function: Callable = None, thresholds_out_of_saliency_map_function: Callable = None
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
    def activations_transformation(self, activations: np.array) -> np.array:
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

        ### XAI ###
        if CUSTOM_HYP.unk.USE_XAI:
            if CUSTOM_HYP.unk.xai.XAI_METHOD == 'D-RISE':
                from yolo_drise.xai.drise import DRISE
                import os
                input_size = (640, 640)
                gpu_batch = 64
                number_of_masks = 6000
                stride = 8
                p1 = 0.5
                expl_model = DRISE(model=model, 
                                  input_size=input_size, 
                                  device=device,
                                  gpu_batch=gpu_batch)
                
                generate_new = False
                mask_file = f"./yolo_drise/masks/masks_640x640.npy"
                if generate_new or not os.path.isfile(mask_file):
                    # explainer.generate_masks(N=5000, s=8, p1=0.1, savepath= mask_file)
                    expl_model.generate_masks(N=number_of_masks, s=stride, p1=p1, savepath=mask_file)
                else:
                    expl_model.load_masks(mask_file)
                    print('Masks are loaded.')
            else:
                # Set model
                expl_model = yolov8_heatmap(
                    weight=model.ckpt_path,
                    method=CUSTOM_HYP.unk.xai.XAI_METHOD,
                    layer=CUSTOM_HYP.unk.xai.XAI_TARGET_LAYERS,
                    ratio=0.05,
                    conf_threshold=self.min_conf_threshold_test,
                    renormalize=CUSTOM_HYP.unk.xai.XAI_RENORMALIZE,
                    show_box=False,
                )
        
        ### Start iterating over the data ###
        c = 0
        for idx_of_batch, data in enumerate(dataloader):
            count_of_images += len(data['im_file'])
            # if count_of_images < 8000:
            #     continue
            ### Preparar imagenes y targets ###
            imgs, targets = self.prepare_data_for_model(data, device)
            
            ### Procesar imagenes en el modelo para obtener logits y las cajas ###
            results = model.predict(imgs, save=False, verbose=True, conf=self.min_conf_threshold_test, device=device)

            ### Comprobar si las cajas predichas son OoD ###
            ood_decision = self.compute_ood_decision_on_results(results, logger)

            ### Añadir posibles cajas desconocidas a las predicciones ###
            if self.enhanced_unk_localization:
                directory_name = f'{now}_{self.name}'
                imgs_folder_path = folder_path / directory_name
                imgs_folder_path.mkdir(exist_ok=True)
                if CUSTOM_HYP.unk.USE_XAI:
                    # TODO: Aqui pruebo lo de la explicabilidad, que tiene que sacar un mapa de valores por imagen, siendo cada mapa una imagen de 80x80
                    #   con los valores de la importancia de cada pixel, que luego se restaran al saliency map correspondiente, escalando los valores al rango
                    #   del saliency map
                    # Crear carpeta
                    directory_name = f'{now}_{self.name}'
                    imgs_folder_path = folder_path / directory_name
                    imgs_folder_path.mkdir(exist_ok=True)
                    # Save hyperparameters as dict
                    from dataclasses import asdict
                    with open(imgs_folder_path / 'hyperparameters.json', 'w') as f:
                        json.dump(asdict(CUSTOM_HYP), f)
                    delattr(model.model, 'which_layers_to_extract')
                    delattr(model.model, 'extraction_mode')
                    # Heatmaps in shape (M, H, W), M being the batch size an in form of a tensor in cpu
                    # and in the range [0, 1]
                    if CUSTOM_HYP.unk.xai.XAI_METHOD == 'D-RISE':
                        from skimage.transform import downscale_local_mean
                        save_name = f'./yolo_drise/heatmaps_{number_of_images_saved}_to_{count_of_images-1}.npy'
                        if not os.path.exists(save_name):
                            expl_heatmaps = expl_model(x=imgs, results=results, mode='object_detection')
                            # Save the heatmaps
                            np.save(save_name, expl_heatmaps.numpy())
                        else:
                            # Load the heatmaps and downscale them
                            print('***** LOADING HEATMAPS *****')
                            expl_heatmaps = torch.tensor(np.load(save_name), dtype=torch.float32)
                        processed_heatmaps = downscale_local_mean(expl_heatmaps.numpy(), (1, 8, 8))
                        processed_heatmaps = [torch.tensor(hm, dtype=torch.float32) for hm in processed_heatmaps]
                    else:
                        expl_heatmaps = expl_model(imgs, return_type='Tensor', show_image=False)
                        processed_heatmaps = expl_heatmaps
                        #processed_heatmaps = limit_heatmaps_to_bounding_boxes(expl_heatmaps, results)                 
                    configure_extra_output_of_the_model(model, self)
                else:
                    processed_heatmaps = None
                if CUSTOM_HYP.unk.RANK_BOXES:
                    possible_unk_bboxes, ood_decision_on_unknown, distances_per_image = self.compute_extra_possible_unkwnown_bboxes_and_decision(
                        results, data, ood_decision_of_results=ood_decision, explainalbility_heatmaps=processed_heatmaps,
                        folder_path=imgs_folder_path, origin_of_idx=idx_of_batch*dataloader.batch_size
                    )
                else:
                    distances_per_image = None
                    possible_unk_bboxes, ood_decision_on_unknown = self.compute_extra_possible_unkwnown_bboxes_and_decision(
                        results, data, ood_decision_of_results=ood_decision, explainalbility_heatmaps=processed_heatmaps,
                        folder_path=imgs_folder_path, origin_of_idx=idx_of_batch*dataloader.batch_size
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
                distances_unk_prop_per_image=distances_per_image
            )
            
            number_of_images_saved += len(data['im_file'])
            # TODO: De momento no queremos plotear todo, solo unos pocos batches
            # if number_of_images_saved > 200:
            #     quit()
            # if idx_of_batch > 10:
            #     quit()
    
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

        # TODO: XAI
        if CUSTOM_HYP.unk.USE_XAI and self.enhanced_unk_localization:
            if CUSTOM_HYP.unk.xai.XAI_METHOD == 'D-RISE':
                from yolo_drise.xai.drise import DRISE
                import os
                input_size = (640, 640)
                gpu_batch = CUSTOM_HYP.unk.xai.drise.GPU_BATCH
                number_of_masks = CUSTOM_HYP.unk.xai.drise.NUMBER_OF_MASKS
                stride = CUSTOM_HYP.unk.xai.drise.STRIDE
                p1 = CUSTOM_HYP.unk.xai.drise.P1
                expl_model = DRISE(model=model, 
                                  input_size=input_size, 
                                  device=device,
                                  gpu_batch=gpu_batch)
                
                generate_new = CUSTOM_HYP.unk.xai.drise.GENERATE_NEW_MASKS
                mask_file = f"./yolo_drise/masks/masks_640x640_{p1:.2f}.npy"
                if generate_new or not os.path.isfile(mask_file):
                    expl_model.generate_masks(N=number_of_masks, s=stride, p1=p1, savepath=mask_file)
                else:
                    expl_model.load_masks(mask_file)
                    print('Masks are loaded.')
            else:
                expl_model = yolov8_heatmap(
                    weight=model.ckpt_path,
                    method=CUSTOM_HYP.unk.xai.XAI_METHOD,
                    layer=CUSTOM_HYP.unk.xai.XAI_TARGET_LAYERS,
                    ratio=0.05,
                    conf_threshold=self.min_conf_threshold_test,
                    renormalize=CUSTOM_HYP.unk.xai.XAI_RENORMALIZE,
                    show_box=False,
                )

        number_of_images_saved = 0
        count_of_images = 0
        for idx_of_batch, data in enumerate(dataloader):
            count_of_images += len(data['im_file'])

            if idx_of_batch % 50 == 0 or idx_of_batch == number_of_batches - 1:
                logger.info(f"{(idx_of_batch/number_of_batches) * 100:02.1f}%: Procesing batch {idx_of_batch+1} of {number_of_batches}") 

            ### Preparar imagenes y targets ###
            imgs, targets = self.prepare_data_for_model(data, device)

            # # Write the file names of the images alongside the number of the image
            # with open(f'./imagenames.txt', 'a') as f:
            #     for i, im_file in enumerate(data['im_file']):
            #         f.write(f'{i+number_of_images_processed} - {im_file}\n')
            # number_of_images_processed += len(data['im_file'])
            # continue
            
            ### Procesar imagenes en el modelo para obtener logits y las cajas ###
            results = model.predict(imgs, save=False, verbose=False, conf=self.min_conf_threshold_test, device=device)

            ### Comprobar si las cajas predichas son OoD ###
            ood_decision = self.compute_ood_decision_on_results(results, logger)

            ### Añadir posibles cajas desconocidas a las predicciones ###
            if self.enhanced_unk_localization:
                if CUSTOM_HYP.unk.USE_XAI:
                    # TODO: Aqui pruebo lo de la explicabilidad, que tiene que sacar un mapa de valores por imagen, siendo cada mapa una imagen de 80x80
                    #   con los valores de la importancia de cada pixel, que luego se restaran al saliency map correspondiente, escalando los valores al rango
                    #   del saliency map
                    # Crear carpeta
                    # directory_name = f'{now}_{self.name}'
                    # imgs_folder_path = folder_path / directory_name
                    # imgs_folder_path.mkdir(exist_ok=True)
                    delattr(model.model, 'which_layers_to_extract')
                    delattr(model.model, 'extraction_mode')
                    # Heatmaps in shape (M, H, W), M being the batch size an in form of a tensor in cpu
                    # and in the range [0, 1]
                    if CUSTOM_HYP.unk.xai.XAI_METHOD == 'D-RISE':
                        from skimage.transform import downscale_local_mean
                        if p1 == 0.5:
                            save_name = f'./yolo_drise/heatmaps_{number_of_images_saved}_to_{count_of_images-1}.npy'
                        else:
                            save_name = f'./yolo_drise/heatmaps_{number_of_images_saved}_to_{count_of_images-1}_p1_{p1:.2f}.npy'
                        if not os.path.exists(save_name):
                            expl_heatmaps = expl_model(x=imgs, results=results, mode='object_detection')
                            # Save the heatmaps
                            np.save(save_name, expl_heatmaps.numpy())
                        else:
                            # Load the heatmaps and downscale them
                            print('***** LOADING HEATMAPS *****')
                            expl_heatmaps = torch.tensor(np.load(save_name), dtype=torch.float32)
                        processed_heatmaps = downscale_local_mean(expl_heatmaps.numpy(), (1, 8, 8))
                        processed_heatmaps = [torch.tensor(hm, dtype=torch.float32) for hm in processed_heatmaps]
                    else:
                        expl_heatmaps = expl_model(imgs, return_type='Tensor', show_image=False)
                        processed_heatmaps = expl_heatmaps
                        #processed_heatmaps = limit_heatmaps_to_bounding_boxes(expl_heatmaps, results)                 
                    configure_extra_output_of_the_model(model, self)
                    
                    # delattr(model.model, 'which_layers_to_extract')
                    # delattr(model.model, 'extraction_mode')
                    # expl_heatmaps = expl_model(imgs, return_type='Tensor', show_image=False)
                    # processed_heatmaps = limit_heatmaps_to_bounding_boxes(expl_heatmaps, results)                 
                    # configure_extra_output_of_the_model(model, self)
                else:
                    processed_heatmaps = None
                if CUSTOM_HYP.unk.RANK_BOXES:
                        possible_unk_bboxes, ood_decision_on_unknown, distances_per_image = self.compute_extra_possible_unkwnown_bboxes_and_decision(
                            results, data, ood_decision_of_results=ood_decision, explainalbility_heatmaps=processed_heatmaps,
                            folder_path=None, origin_of_idx=idx_of_batch*dataloader.batch_size
                        )
                else:
                    distances_per_image = None
                    possible_unk_bboxes, ood_decision_on_unknown = self.compute_extra_possible_unkwnown_bboxes_and_decision(
                        results, data, ood_decision_of_results=ood_decision, explainalbility_heatmaps=processed_heatmaps,
                        folder_path=None, origin_of_idx=idx_of_batch*dataloader.batch_size
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

        sufficient_samples = CUSTOM_HYP.SUFFICIENT_NUM_SAMPLES
        good_number_of_samples = CUSTOM_HYP.GOOD_NUM_SAMPLES

        # One threshold per class
        if self.per_class:
            
            # One threshold per stride
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
                            if idx_cls < 20:
                                logger.warning(f'Class {idx_cls:03}, Stride {idx_stride} -> Has less than {sufficient_samples} samples. No threshold is generated')
            
            # Same threshold for all strides
            else:
                # Per class with no stride differentiation
                thresholds = [0 for _ in range(len(ind_scores))]
                for idx_cls, cl_scores in enumerate(ind_scores):
                    if len(cl_scores) > sufficient_samples:
                        thresholds[idx_cls] = float(np.percentile(cl_scores, used_tpr, method='lower'))
                        if len(cl_scores) < good_number_of_samples:
                            logger.warning(f"Class {idx_cls}: {len(cl_scores)} samples. The threshold may not be accurate")
                    else:
                        if idx_cls < 20:
                            logger.warning(f"Class {idx_cls} has less than {sufficient_samples} samples. No threshold is generated")

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
            explainalbility_heatmaps: Optional[List[Tensor]] = None,
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
                import matplotlib.pyplot as plt
                from skimage.transform import resize
                plt.imshow(saliency_map, cmap='viridis')
                plt.colorbar()
                plt.savefig(folder_path / f'{(origin_of_idx + img_idx):03}_saliency_map.{IMAGE_FORMAT}')
                plt.close()
                saliency_map_plot = resize(saliency_map, (data['img'][img_idx].shape[2], data['img'][img_idx].shape[1]))
                saliency_map_plot = saliency_map_plot - saliency_map.min()
                saliency_map_plot = saliency_map_plot / saliency_map_plot.max()
                saliency_map_plot = (saliency_map_plot * 255).astype(np.uint8)
                plt.imshow(data['img'][img_idx].permute(1, 2, 0).cpu().numpy())
                plt.imshow(saliency_map_plot, cmap='viridis', alpha=0.5)
                plt.savefig(folder_path / f'{(origin_of_idx + img_idx):03}_saliency_map_over_image.{IMAGE_FORMAT}')
                plt.close()

            if explainalbility_heatmaps:
                # Only use the heatmap if it is not 0 (checked using the sum)
                heatmap = explainalbility_heatmaps[img_idx]
                if heatmap is not None:
                    if heatmap.sum() > 0:
                        # Remove padding from heatmaps
                        x_padding, y_padding = padding_x_y
                        htmap_h, htmap_w = heatmap.shape[0], heatmap.shape[1]
                        heatmap = heatmap[y_padding:htmap_h-y_padding, x_padding:htmap_w-x_padding]
                        heatmap_for_plot = heatmap.clone().numpy()
                        if CUSTOM_HYP.unk.USE_XAI_TO_MODIFY_SALIENCY:
                            if CUSTOM_HYP.unk.xai.INFO_MERGING_METHOD == "scale_then_minus":
                                # Scale the heatmap [0, 1] to the values of the saliency map
                                heatmap = (saliency_map.max() - saliency_map.min()) * (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) + saliency_map.min()
                                heatmap = heatmap.numpy()
                                saliency_map = saliency_map - heatmap
                                saliency_map = np.clip(saliency_map, 0, saliency_map.max())
                            elif CUSTOM_HYP.unk.xai.INFO_MERGING_METHOD == "multiply":
                                # Take the element-wise multiplication of the saliency map and the heatmap,
                                # using the heatmap values as weights but inverted (1-htmap_value) * saliency_map
                                heatmap_inverted = 1 - heatmap.numpy()
                                saliency_map = saliency_map * heatmap_inverted
                                if folder_path:
                                    plt.imshow(heatmap_inverted, cmap='viridis')
                                    plt.colorbar()
                                    plt.savefig(folder_path / f'{(origin_of_idx + img_idx):03}_heatmap_inverted.{IMAGE_FORMAT}')
                                    plt.close()
                            elif CUSTOM_HYP.unk.xai.INFO_MERGING_METHOD == "turn_off_pixels":
                                # Put a threshold on the heatmap and turn off the pixels that are above the threshold
                                threshold = 0.5
                                heatmap_thresholded = heatmap.numpy() < threshold  # Turn off the pixels above the threshold
                                heatmap_thresholded = heatmap_thresholded.astype(int)
                                saliency_map = heatmap_thresholded * saliency_map
                                if folder_path:
                                    plt.imshow(heatmap_thresholded, cmap='gray')
                                    plt.savefig(folder_path / f'{(origin_of_idx + img_idx):03}_heatmap_thresholded.{IMAGE_FORMAT}')
                                    plt.close()
                            elif CUSTOM_HYP.unk.xai.INFO_MERGING_METHOD == "sigmoid":
                                # Use the heatmap as a mask for the saliency map
                                heatmap = torch.sigmoid((heatmap - CUSTOM_HYP.unk.xai.SIGMOID_INTERCEPT) * CUSTOM_HYP.unk.xai.SIGMOID_SLOPE)
                                heatmap = 1 - heatmap.numpy()  # Invert the values
                                saliency_map = heatmap * saliency_map
                                if folder_path:
                                    plt.imshow(heatmap, cmap='viridis')
                                    plt.colorbar()
                                    plt.savefig(folder_path / f'{(origin_of_idx + img_idx):03}_heatmap_sigmoid.{IMAGE_FORMAT}')
                                    plt.close()
                            else:
                                raise ValueError("The method to merge the saliency map and the heatmap is not valid")
                        # Plots
                        if folder_path:
                            # Save the heatmap
                            plt.imshow(heatmap_for_plot, cmap='viridis')
                            plt.colorbar()
                            plt.savefig(folder_path / f'{(origin_of_idx + img_idx):03}_heatmap_original.{IMAGE_FORMAT}')
                            plt.close()
                            # Save the heatmap over the image
                            #[:, y_padding:ftmap_height-y_padding, x_padding:ftmap_width-x_padding]
                            orig_h, orig_w = data['img'][img_idx].shape[1:]
                            orig_pd_x, orig_pd_y = original_img_padding_x_y.astype(int)
                            orig_img_no_pad = data['img'][img_idx][:, orig_pd_y:orig_h-orig_pd_y, orig_pd_x:orig_w-orig_pd_x]
                            plt.imshow(orig_img_no_pad.permute(1, 2, 0).cpu().numpy()/ 255)
                            plt.imshow(resize(heatmap_for_plot, orig_img_no_pad.shape[1:]), cmap='viridis', alpha=0.5)
                            plt.savefig(folder_path / f'{(origin_of_idx + img_idx):03}_heatmap_original_over_image.{IMAGE_FORMAT}')
                            plt.close()
                            if CUSTOM_HYP.unk.USE_XAI_TO_MODIFY_SALIENCY:
                                # Save the saliency map
                                plt.imshow(saliency_map, cmap='viridis')
                                plt.colorbar()
                                plt.savefig(folder_path / f'{(origin_of_idx + img_idx):03}_saliency_map_new.{IMAGE_FORMAT}')
                                plt.close()

            ### 3. Compute the thresholds to binarize the saliency map
            thresholds = self.compute_thresholds_out_of_saliency_map(saliency_map)

            if folder_path:  # Save the thresholded images in one figure
                fig, axs = plt.subplots(1, len(thresholds), figsize=(5*len(thresholds), 5))
                for idx, thr in enumerate(thresholds):
                    axs[idx].imshow(saliency_map > thr, cmap='gray')
                    axs[idx].set_title(f'Thr: {thr:.2f}')
                plt.savefig(folder_path / f'{(origin_of_idx + img_idx):03}_thresholded_saliency_map.{IMAGE_FORMAT}')
                plt.close()
                
            ### 4. Extract the bounding boxes from the saliency map using the thresholds
            possible_unk_boxes_per_thr = extract_bboxes_from_saliency_map_and_thresholds(saliency_map, thresholds)

            if CUSTOM_HYP.unk.USE_XAI_TO_REMOVE_PROPOSALS:
                heatmap_boxes = None
                if heatmap is not None and not heatmap.isnan().any():
                    # Make the same process of obtaining boxes with the heatmap
                    thresholds_heatmap = self.compute_thresholds_out_of_saliency_map(heatmap.numpy())
                    heatmap_boxes_per_thr = extract_bboxes_from_saliency_map_and_thresholds(heatmap, thresholds_heatmap)
                    heatmap_boxes = torch.cat(heatmap_boxes_per_thr, dim=0)
                    if folder_path:
                        # Plot the heatmap with the boxes
                        from torchvision.utils import draw_bounding_boxes
                        im = draw_bounding_boxes(image=(heatmap*255).to(torch.uint8).unsqueeze(0), boxes=heatmap_boxes)
                        plt.imshow(im.permute(1, 2, 0).cpu().numpy())
                        plt.savefig(folder_path / f'{(origin_of_idx + img_idx):03}_heatmap_with_boxes.{IMAGE_FORMAT}')
                        plt.close()
                        # Plot the thresholded heatmap
                        fig, axs = plt.subplots(1, len(thresholds_heatmap), figsize=(5*len(thresholds_heatmap), 5))
                        for idx, thr in enumerate(thresholds_heatmap):
                            axs[idx].imshow(heatmap > thr, cmap='gray')
                            axs[idx].set_title(f'Thr: {thr:.2f}')
                        plt.savefig(folder_path / f'{(origin_of_idx + img_idx):03}_thresholded_heatmap.{IMAGE_FORMAT}')
                        plt.close()
                else:
                    print(f"Image {img_idx} has no heatmap")
                    
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
                xai_boxes=heatmap_boxes if CUSTOM_HYP.unk.USE_XAI_TO_REMOVE_PROPOSALS else None,
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
                               selected_stride: int, xai_boxes: Optional[Tensor] = None, folder_str: Optional[str] = None) -> Tensor:
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
        # Add the padding to the XAI boxes if they are provided
        if xai_boxes is not None:
            xai_boxes[:, 0] += padding[0]
            xai_boxes[:, 1] += padding[1]
            xai_boxes[:, 2] += padding[0]
            xai_boxes[:, 3] += padding[1]
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
            # Use pred boxes to remove some unk_proposals_one_thr
            if len(bbox_preds_in_ftmap_size) > 0:
                if CUSTOM_HYP.unk.MAX_IOU_WITH_PREDS > 0:
                    # 3º: Remove unk_proposals_one_thr with IoU > iou_thr with the predictions
                    # Compute the IoU with the predictions
                    ious = box_iou(unk_proposals_one_thr, bbox_preds_in_ftmap_size)
                    # Remove the unk_proposals_one_thr with IoU > iou_thr
                    mask = ious.max(dim=1).values < CUSTOM_HYP.unk.MAX_IOU_WITH_PREDS
                    unk_proposals_one_thr = unk_proposals_one_thr[mask]

                # # Plot Unk prop (in yellow) and preds (in green) 
                # import matplotlib.pyplot as plt
                # from torchvision.utils import draw_bounding_boxes
                # # Convert to tensor
                # unk = unk_proposals_one_thr.to(torch.uint8)
                # pred = bbox_preds_in_ftmap_size.to(torch.uint8)
                # # Create the image
                # img = torch.ones((3, ftmap_height + padding[1]*2, ftmap_width + padding[0]*2), dtype=torch.uint8)
                # # Merge the bouding boxes and create the colors for each
                # boxes = torch.cat([unk, pred], dim=0)
                # colors = ["yellow"] * len(unk) + ["green"] * len(pred)
                # # Draw the boxes
                # img = draw_bounding_boxes(img, boxes, colors=colors)
                # plt.imshow(img.permute(1, 2, 0).cpu().numpy())
                # plt.savefig('AAA_unk_prop_and_preds.png')
                # plt.close()

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
                
                if CUSTOM_HYP.unk.USE_XAI_TO_REMOVE_PROPOSALS:
                    #assert xai_boxes is not None, "You must provide the XAI boxes to remove the proposals"
                    # 5º: Remove unk_proposals_one_thr that are close to the XAI boxes
                    if len(unk_proposals_one_thr) > 0 and xai_boxes is not None:
                        # Compute the IoU with the XAI boxes
                        ious = box_iou(unk_proposals_one_thr, xai_boxes)
                        # Remove the unk_proposals_one_thr with IoU > iou_thr
                        mask = ious.max(dim=1).values < CUSTOM_HYP.unk.xai.MAX_IOU_WITH_XAI
                        # Plot the unk_proposals in yellow and the XAI boxes in red
                        import matplotlib.pyplot as plt
                        from torchvision.utils import draw_bounding_boxes
                        # Convert to tensor
                        unk = unk_proposals_one_thr.to(torch.uint8)
                        xai = xai_boxes.to(torch.uint8)
                        # Create the image
                        img = torch.ones((1, unpaded_ftmap_height + padding[1]*2, unpaded_ftmap_width + padding[0]*2), dtype=torch.uint8)
                        # Merge the bouding boxes and create the colors for each
                        boxes = torch.cat([unk, xai], dim=0)
                        colors = ["yellow"] * len(unk) + ["red"] * len(xai)
                        # Draw the boxes
                        img = draw_bounding_boxes(img, boxes, colors=colors)
                        plt.imshow(img.permute(1, 2, 0).cpu().numpy())
                        plt.savefig(f'{folder_str}_heatmap_and_proposals_thr{idx_thr}.png')
                        plt.close()

                        unk_proposals_one_thr = unk_proposals_one_thr[mask]

            #### Ranking boxes using diferent methods ####
            if CUSTOM_HYP.unk.RANK_BOXES:
                if len(unk_proposals_one_thr) > 0:
                    # 5º: Rank the unk_proposals_one_thr using their features (extracted from the feature maps) and comparing them
                    #   to the centroids of every known class. Then, select the unk_proposals_one_thr that are far from the centroids
                    if False:
                        # OPT 1:
                        features_per_proposal = []
                        for box in unk_proposals_one_thr:
                            x1, y1, x2, y2 = box
                            # Extract the feature map region corresponding to the bounding box
                            extracted_features = paded_feature_maps[:, y1:y2, x1:x2]
                            # Optionally, you might want to apply some pooling or other operation to standardize the size
                            extracted_features = F.adaptive_avg_pool2d(extracted_features, (1, 1))
                            features_per_proposal.append(extracted_features)
                        # Optionally convert list to a tensor
                        features_per_proposal = torch.stack(features_per_proposal)
                    else:
                        # OPT 2: use roi align
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
                                self.activations_transformation(features_per_proposal)
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
                        # Compute the entropy using scipy
                        from scipy.stats import entropy
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
                all_distances_per_proposal = np.array([])

            if CUSTOM_HYP.unk.rank.MAX_NUM_UNK_BOXES_PER_IMAGE > 0 and len(all_distances_per_proposal) > 0:
                if CUSTOM_HYP.unk.rank.NMS > 0:
                    # Apply NMS to the unk_proposals
                    from torchvision.ops import nms
                    # Returns the indices of the boxes that we want to keep in DESCENDING order of scores
                    if CUSTOM_HYP.unk.rank.GET_BOXES_WITH_GREATER_RANK:
                        keep = nms(all_unk_prop, torch.from_numpy(all_distances_per_proposal), iou_threshold=CUSTOM_HYP.unk.rank.NMS)
                    else:
                        keep = nms(all_unk_prop, torch.from_numpy(-all_distances_per_proposal), iou_threshold=CUSTOM_HYP.unk.rank.NMS)
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
                 min_conf_threshold_train: float, min_conf_threshold_test: float, **kwargs):
        is_distance_method = False
        which_internal_activations = 'logits'  # Always logits for these methods
        enhanced_unk_localization = False  # By default not used with logits, as feature maps are needed.
        super().__init__(name, is_distance_method, per_class, per_stride, iou_threshold_for_matching,
                         min_conf_threshold_train, min_conf_threshold_test, which_internal_activations, enhanced_unk_localization)
        self.cluster_method = 'None'

    def compute_ood_decision_on_results(self, results: Results, logger: Logger) -> List[List[int]]:
        ood_decision = []  
        for idx_img, res in enumerate(results):
            ood_decision.append([])  # Every image has a list of decisions for each bbox
            for idx_bbox in range(len(res.boxes.cls)):
                cls_idx = int(res.boxes.cls[idx_bbox].cpu())
                logits = res.extra_item[idx_bbox][4:].cpu()
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
                logits = res.extra_item[idx_bbox][4:].cpu()
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
            elif score < self.min_conf_threshold_train:
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
            
            return a * score + b
        
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
        # x = np.linspace(0, self.thresholds[cls_idx], 100)
        # y = a * x + b
        # plt.plot(x, y)
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
                logits_one_bbox = res.extra_item[valid_idx_one_bbox][4:].cpu()
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
                    scores[idx_cls] = self.compute_scores(activations[idx_cls], idx_cls)
                else:
                    scores[idx_cls] = np.array([])
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

    def activations_transformation(self, activations: np.array) -> np.array:
        raise NotImplementedError("This method is not needed for methods using logits")

    def compute_distance(self, centroids: np.array, features: np.array) -> np.array:
        raise NotImplementedError("This method is not needed for methods using logits")


class MSP(LogitsMethod):

    def __init__(self, **kwargs):
        name = 'MSP'
        super().__init__(name, **kwargs)
    
    # def compute_score_one_bbox(self, logits: Tensor, cls_idx: int) -> float:
    #     logits = logits.numpy()
    #     assert cls_idx == logits.argmax(), "The max logit is not the one of the predicted class"
    #     return logits[cls_idx]
    
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
    
    # def compute_score_one_bbox(self, logits, cls_idx) -> float:
    #     if len(logits.shape) == 1:
    #         return self.temper * torch.logsumexp(logits.unsqueeze(0) / self.temper, dim=0)
    #     return self.temper * torch.logsumexp(logits / self.temper, dim=0).item()
    
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
    metric: 'str'

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
                    img_shape=res.orig_img.shape[2:],
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
                            self.activations_transformation(ftmap)
                        )[0]

                    ood_decision.append(self.compute_indness(distance, cls_idx))


    def compute_indness(self, score: float, cls_idx: int, stride_idx: int) -> float:
        """
        Compute an score between -1 and 1. -1 means that the score is the maximum OOD score, 1 means that is the maximum IND score.
        In logits methods, it will be based on a linear function between the threshold and the minimum and maximum score.
        The maximum is 1, as a logit cannot be greater than 1. The minimum is the minimum confidence for the predictions.
        """
        if CUSTOM_HYP.fusion.DISTANCE_USE_FROM_ZERO_TO_THR:
        
            a = -1 / (self.thresholds[cls_idx] - 1)
            b = 1 - a

        elif CUSTOM_HYP.fusion.DISTANCE_USE_IN_DISTRIBUTION_TO_DEFINE_LIMITS:
            if self.per_class and self.per_stride:
                # Is a piecewise function
                if score > self.thresholds[cls_idx][stride_idx]:
                    a = -1 / (self.max_dist[cls_idx][stride_idx] - self.thresholds[cls_idx][stride_idx])
                    b = self.thresholds[cls_idx][stride_idx] / (self.max_dist[cls_idx][stride_idx] - self.thresholds[cls_idx][stride_idx])
                elif score < self.min_conf_threshold_train:
                    a = 1 / (self.min_dist[cls_idx][stride_idx] - self.thresholds[cls_idx][stride_idx])
                    b = -self.thresholds[cls_idx][stride_idx] / (self.min_dist[cls_idx][stride_idx] - self.thresholds[cls_idx][stride_idx])
                else:
                    print("Score is equal to the threshold")
                    a = 0
                    b = 0
                
                return a * score + b

            else:
                raise NotImplementedError("Not implemented yet")

        
        return a * score + b
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
                    )
                    
                    # Add the valid roi aligned ftmaps to the list. As we only introduce one image, we need to get the only element of roi aligned ftmaps
                    self._extract_valid_preds_from_one_image_roi_aligned_ftmaps(cls_idx_one_pred, roi_aligned_ftmaps_per_stride[0], valid_preds, all_activations)

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
                        activations_one_cls_one_stride_transformed = self.activations_transformation(activations_one_cls_one_stride)
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
                            #self.clusters[idx_cls][idx_stride][None, :],
                            self.clusters[idx_cls][idx_stride],
                            self.activations_transformation(activations_one_cls_one_stride)
                            # activations_one_cls_one_stride.reshape(activations_one_cls_one_stride.shape[0], -1)
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
                    img_shape=res.orig_img.shape[2:],
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

            self._compute_ood_decision_for_one_result_from_roi_aligned_feature_maps(
                idx_img=idx_img,
                one_img_bboxes_cls_idx=res.boxes.cls.cpu(),
                roi_aligned_ftmaps_one_img_per_stride=roi_aligned_ftmaps_per_stride,  # As we are processing one image only
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
                            self.activations_transformation(ftmap)
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
                                #cluster[idx_stride][None, :],
                                cluster[idx_stride],
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
                                #self.clusters[_cls_idx][idx_stride][None, :],
                                self.clusters[_cls_idx][idx_stride],
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

                elif self.cluster_method == 'all':
                    raise NotImplementedError("As the amount of In-Distribution data is too big," \
                                            "ir would be intractable to treat each sample as a cluster")

                elif self.cluster_method in AVAILABLE_CLUSTERING_METHODS:
                    self.generate_multiple_cluster_per_class_per_stride(ind_tensors, clusters_per_class_and_stride, logger)

                else:
                    raise NameError(f"The clustering_opt must be one of the following: 'one', 'all', or one of {AVAILABLE_CLUSTERING_METHODS}." \
                                    f"Current value: {self.cluster_method}")
                
            else:
                raise NotImplementedError("Not implemented yet")
            
        else:
            raise NotImplementedError("Not implemented yet")

        return clusters_per_class_and_stride
    
    def generate_one_cluster_per_class_and_stride(self, ind_tensors: List[List[np.ndarray]], clusters_per_class_and_stride: List[List[np.ndarray]], logger):
        for idx_cls, ftmaps_one_cls in enumerate(ind_tensors):

            #logger.info(f'Class {idx_cls:03} of {len(ind_tensors)}')
            for idx_stride, ftmaps_one_cls_one_stride in enumerate(ftmaps_one_cls):
                
                if len(ftmaps_one_cls_one_stride) > 1:
                    ftmaps_one_cls_one_stride = self.activations_transformation(ftmaps_one_cls_one_stride)
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

        if CUSTOM_HYP.clusters.VISUALIZE:
            import matplotlib.pyplot as plt
            from collections import Counter
            folder_for_hist = Path('figures/histograms')
            folder_for_hist.mkdir(parents=False, exist_ok=True)
            folder_for_cluster_scores = Path('figures/cluster_scores')
            folder_for_cluster_scores.mkdir(parents=False, exist_ok=True)

        for idx_cls, ftmaps_one_cls in enumerate(ind_tensors):
            #logger.info(f'Class {idx_cls:03} of {len(ind_tensors)}')
            cluster_indices_one_class = []
            for idx_stride, ftmaps_one_cls_one_stride in enumerate(ftmaps_one_cls):
                
                if len(ftmaps_one_cls_one_stride) > 1:
                    ftmaps_one_cls_one_stride = self.activations_transformation(ftmaps_one_cls_one_stride)
                    # 1. Find the optimal number of clusters and obtain the labels
                    if CUSTOM_HYP.clusters.VISUALIZE:
                        string_for_visualization = f'class{idx_cls:03}_stride{idx_stride}'
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
                    )
                    
                    # Obtain the indices of the clusters
                    cluster_indices = set(cluster_labels)
                    cluster_indices_one_class.append(cluster_indices)

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
                    cluster_indices_one_class.append([])

            if CUSTOM_HYP.clusters.VISUALIZE:
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle(f'Class {idx_cls:03}')
                for _i, cl_lbl in enumerate(cluster_indices_one_class):
                    # Count the number of samples per cluster
                    cluster_counts = Counter(cl_lbl)
                    # Extracting cluster indices and their respective counts
                    one_stride_labels = list(cluster_counts.keys())
                    one_stride_counts = list(cluster_counts.values())
                    if len(cl_lbl) > 0:
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
                plt.tight_layout()
                plt.savefig(folder_for_hist / f'histogram_{self.cluster_method}_{idx_cls:03}_{self.cluster_optimization_metric}.png')
                # plt.hist(cluster_labels, bins=len(cluster_indices))
                # plt.savefig(folder_for_hist / f'histogram_{self.cluster_method}_{idx_cls:03}_{idx_stride}_{self.cluster_optimization_metric}.png')
                # plt.savefig(folder_for_hist / f'histogram_{self.cluster_method}_{idx_cls:03}_{idx_stride}_{self.cluster_optimization_metric}.png')
                plt.close()

    def activations_transformation(self, activations: np.array) -> np.array:
        """
        Transform the activations to the shape needed to compute the distance.
        By default, it flattens the activations leaving the batch dimension as the first dimension.
        """
        return activations.reshape(activations.shape[0], -1)


class _PairwiseDistanceClustersPerClassPerStride(DistanceMethod):
    def __init__(self, name: str, metric: str, **kwargs):
            AVAILABLE_PAIRWISE_METRICS = ['cosine', 'l1', 'l2']
            per_class = True
            per_stride = True
            #cluster_optimization_metric = 'silhouette'
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


# class L1DistanceOneClusterPerStride(DistanceMethod):
    
#         # name: str, agg_method: str, per_class: bool, per_stride: bool, cluster_method: str, cluster_optimization_metric: str
#         def __init__(self, **kwargs):
#             name = 'L1DistancePerStride'
#             per_class = True
#             per_stride = True
#             cluster_method = 'one'
#             cluster_optimization_metric = 'silhouette'
#             super().__init__(name, per_class, per_stride, cluster_method, cluster_optimization_metric, **kwargs)
        
#         def compute_distance(self, cluster: np.array, activations: np.array) -> List[float]:

#             distances = pairwise_distances(
#                 cluster,
#                 activations,
#                 metric='l1'
#                 )

#             #return distances[0]
#             return distances.min(axis=0)
        

# class L2DistanceOneClusterPerStride(DistanceMethod):
    
#         # name: str, agg_method: str, per_class: bool, per_stride: bool, cluster_method: str, cluster_optimization_metric: str
#         def __init__(self,  **kwargs):
#             name = 'L2DistancePerStride'
#             per_class = True
#             per_stride = True
#             cluster_method = 'one'
#             cluster_optimization_metric = 'silhouette'
#             super().__init__(name, per_class, per_stride, cluster_method, cluster_optimization_metric, **kwargs)
        
#         def compute_distance(self, cluster: np.array, activations: np.array) -> List[float]:

#             distances = pairwise_distances(
#                 cluster,
#                 activations,
#                 metric='l2'
#                 )

#             #return distances[0]
#             return distances.min(axis=0)

# class CosineDistanceOneClusterPerStride(DistanceMethod):
    
#         # name: str, agg_method: str, per_class: bool, per_stride: bool, cluster_method: str, cluster_optimization_metric: str
#         #def __init__(self, agg_method, **kwargs):
#         def __init__(self, **kwargs):
#             self.metric = 'cosine'
#             name = 'CosineDistancePerStride'
#             per_class = True
#             per_stride = True
#             #cluster_method = 'one'
#             cluster_optimization_metric = 'silhouette'
#             #super().__init__(name, per_class, per_stride, cluster_method, cluster_optimization_metric, **kwargs)
#             super().__init__(name, per_class, per_stride, cluster_optimization_metric=cluster_optimization_metric, **kwargs)
        
#         def compute_distance(self, cluster: np.array, activations: np.array) -> List[float]:
#             distances = pairwise_distances(
#                 cluster,
#                 activations,
#                 metric='cosine'
#                 )
#             #return distances[0]
#             return distances.min(axis=0)
        

class GAPL2DistanceOneClusterPerStride(DistanceMethod):
    
        # name: str, agg_method: str, per_class: bool, per_stride: bool, cluster_method: str, cluster_optimization_metric: str
        def __init__(self,  **kwargs):
            name = 'GAP_L2DistancePerStride'
            per_class = True
            per_stride = True
            cluster_method = 'one'
            #cluster_optimization_metric = 'silhouette'
            super().__init__(name, per_class, per_stride, cluster_method, **kwargs)
        
        def compute_distance(self, cluster: np.array, activations: np.array) -> List[float]:

            distances = pairwise_distances(
                cluster,
                activations,
                metric='l2'
                )

            #return distances[0]
            return distances.min(axis=0)
        
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
        #cluster_optimization_metric = 'silhouette'
        super().__init__(name, per_class, per_stride, cluster_method, **kwargs)

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
        #cluster_optimization_metric = 'silhouette'
        super().__init__(name, per_class, per_stride, cluster_method, **kwargs)

    def compute_distance(self, cluster: np.array, activations: np.array) -> List[float]:
        raise NotImplementedError("Not implemented yet")
    
    def activations_transformation(self, activations: np.array) -> np.array:
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

    logits_method: LogitsMethod
    distance_method: DistanceMethod
    fusion_strategy: str

    def __init__(self, logits_method: LogitsMethod, distance_method: DistanceMethod, fusion_strategy: str,  **kwargs):

        name = f'fusion_{logits_method.name}_{distance_method.name}'
        self.logits_method = logits_method
        self.distance_method = distance_method
        self.fusion_strategy = fusion_strategy
        super().__init__(name=name, per_class=True, per_stride=True, is_distance_method=True, which_internal_activations="none", **kwargs)

        # Define as properties the clusters and thresholds
        self._clusters = None
        self._thresholds = None
    
    # Clusters
    @property
    def clusters(self):
        return self.distance_method.clusters
    
    @clusters.setter
    def clusters(self, clusters):
        self.distance_method.clusters = clusters

    # Thresholds
    @property
    def thresholds(self):
        return self.logits_method.thresholds, self.distance_method.thresholds
    
    @thresholds.setter
    def thresholds(self, thresholds):
        if thresholds is not None:
            if len(thresholds) == 2:
                self.logits_method.thresholds = thresholds[0]
                self.distance_method.thresholds = thresholds[1]
            else:
                raise ValueError("The thresholds must be a tuple with two elements, one for the logits and one for the distance")
        else:
            print("The thresholds must be a tuple with two elements, one for the logits and one for the distance")
            self.logits_method.thresholds = None
            self.distance_method.thresholds = None


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

    def activations_transformation(self, activations: np.array) -> np.array:
        raise NotImplementedError("This method is not going to be called directly")

    def compute_distance(self, centroids: np.array, features: np.array) -> np.array:
        raise NotImplementedError("This method is not going to be called directly")

    def iterate_data_to_extract_ind_activations(self, data_loader, model: YOLO, device: str, logger: Logger):
        
        configure_extra_output_of_the_model(model, which_internal_activations=self.logits_method.which_internal_activations)
        all_activations_logits = self.logits_method.iterate_data_to_extract_ind_activations(data_loader, model, device, logger)
        configure_extra_output_of_the_model(model, which_internal_activations=self.distance_method.which_internal_activations)
        all_activations_ftmaps = self.distance_method.iterate_data_to_extract_ind_activations(data_loader, model, device, logger)

        return all_activations_logits, all_activations_ftmaps

    def generate_thresholds(self, ind_scores: list, tpr: float, logger: Logger) -> Union[List[float], List[List[float]]]:
        
        logits, ftmaps = ind_scores
        logits_thrs = self.logits_method.generate_thresholds(logits, tpr, logger)
        distance_thrs = self.distance_method.generate_thresholds(ftmaps, tpr, logger)

        return logits_thrs, distance_thrs

    def generate_clusters(self, ind_tensors: Union[List[np.ndarray], List[List[np.ndarray]]], logger: Logger) -> Union[List[np.ndarray], List[List[np.ndarray]]]:
        ftmaps = ind_tensors[1]
        return self.distance_method.generate_clusters(ftmaps, logger)
    
    def compute_scores_from_activations(self, activations: Union[List[np.ndarray], List[List[np.ndarray]]], logger: Logger) -> Tuple[List[List[float]], List[List[float]]]:
        
        logits, ftmaps = activations
        logit_scores = self.logits_method.compute_scores_from_activations(logits, logger)
        distance_scores = self.distance_method.compute_scores_from_activations(ftmaps, logger)

        return logit_scores, distance_scores

    def fuse_ood_decisions(self, ood_decision_logits: List[List[int]], ood_decision_distance: List[List[int]]) -> List[List[int]]:
        # 1 is InD, 0 is OoD
        ood_decision = []
        if self.fusion_strategy == "and":
            # AND strategy: If one of methods say that the bbox is InD (decision = 1), then it is InD
            for idx_img in range(len(ood_decision_logits)):
                ood_decision.append(
                    [max(ood_decision_logits[idx_img][idx_bbox], ood_decision_distance[idx_img][idx_bbox]) for idx_bbox in range(len(ood_decision_logits[idx_img]))]
                )

        elif self.fusion_strategy == "or":
            # OR strategy: If one of the methods says that the bbox is OoD (decision = 0), then it is OoD
            for idx_img in range(len(ood_decision_logits)):
                ood_decision.append(
                    [min(ood_decision_logits[idx_img][idx_bbox], ood_decision_distance[idx_img][idx_bbox]) for idx_bbox in range(len(ood_decision_logits[idx_img]))]
                )
        
        elif self.fusion_strategy == "score":
            # SCORE strategy: The score is the sum of the scores of the two methods. If the score is greater than 0, it is InD
            for idx_img in range(len(ood_decision_logits)):
                ood_score_one_img = [ood_decision_logits[idx_img][idx_bbox] + ood_decision_distance[idx_img][idx_bbox] for idx_bbox in range(len(ood_decision_logits[idx_img]))]
                ood_decision.append(
                    [1 if score > 0 else 0 for score in ood_score_one_img]                    
                )
            
        else:
            raise NotImplementedError("Not implemented yet")
        
        # Assert that the number of bboxes is the same after the decision
        for idx_img in range(len(ood_decision)):
            assert len(ood_decision[idx_img]) == len(ood_decision_logits[idx_img]), "The number of bboxes is different"
            assert len(ood_decision[idx_img]) == len(ood_decision_distance[idx_img]), "The number of bboxes is different"

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

        # TODO: XAI
        if CUSTOM_HYP.unk.USE_XAI and self.enhanced_unk_localization:
            if CUSTOM_HYP.unk.xai.XAI_METHOD == 'D-RISE':
                from yolo_drise.xai.drise import DRISE
                import os
                input_size = (640, 640)
                gpu_batch = CUSTOM_HYP.unk.xai.drise.GPU_BATCH
                number_of_masks = CUSTOM_HYP.unk.xai.drise.NUMBER_OF_MASKS
                stride = CUSTOM_HYP.unk.xai.drise.STRIDE
                p1 = CUSTOM_HYP.unk.xai.drise.P1
                expl_model = DRISE(model=model, 
                                  input_size=input_size, 
                                  device=device,
                                  gpu_batch=gpu_batch)
                
                generate_new = CUSTOM_HYP.unk.xai.drise.GENERATE_NEW_MASKS
                mask_file = f"./yolo_drise/masks/masks_640x640_{p1:.2f}.npy"
                if generate_new or not os.path.isfile(mask_file):
                    expl_model.generate_masks(N=number_of_masks, s=stride, p1=p1, savepath=mask_file)
                else:
                    expl_model.load_masks(mask_file)
                    print('Masks are loaded.')
            else:
                expl_model = yolov8_heatmap(
                    weight=model.ckpt_path,
                    method=CUSTOM_HYP.unk.xai.XAI_METHOD,
                    layer=CUSTOM_HYP.unk.xai.XAI_TARGET_LAYERS,
                    ratio=0.05,
                    conf_threshold=self.min_conf_threshold_test,
                    renormalize=CUSTOM_HYP.unk.xai.XAI_RENORMALIZE,
                    show_box=False,
                )

        number_of_images_saved = 0
        count_of_images = 0
        for idx_of_batch, data in enumerate(dataloader):
            count_of_images += len(data['im_file'])

            if idx_of_batch % 50 == 0 or idx_of_batch == number_of_batches - 1:
                logger.info(f"{(idx_of_batch/number_of_batches) * 100:02.1f}%: Procesing batch {idx_of_batch+1} of {number_of_batches}") 

            ### Preparar imagenes y targets ###
            imgs, targets = self.prepare_data_for_model(data, device)

            ###
            # Logits method
            ###
            configure_extra_output_of_the_model(model, ood_method=self.logits_method)
            ### Procesar imagenes en el modelo para obtener logits y las cajas ###
            results_logits = model.predict(imgs, save=False, verbose=False, conf=self.min_conf_threshold_test, device=device)
            ### Comprobar si las cajas predichas son OoD ###
            if self.fusion_strategy == 'score':
                ood_decision_logits = self.logits_method.compute_INDness_scores_on_results(results_logits, logger)
            else:
                ood_decision_logits = self.logits_method.compute_ood_decision_on_results(results_logits, logger)

            ###
            # Distance method
            ###
            configure_extra_output_of_the_model(model, ood_method=self.distance_method)
            ### Procesar imagenes en el modelo para obtener las caracteristicas y las cajas ###
            results_distance = model.predict(imgs, save=False, verbose=False, conf=self.min_conf_threshold_test, device=device)
            ### Comprobar si las cajas predichas son OoD ###
            if self.fusion_strategy == 'score':
                ood_decision_distance = self.distance_method.compute_INDness_scores_on_results(results_distance, logger)
            else:
                ood_decision_distance = self.distance_method.compute_ood_decision_on_results(results_distance, logger)

            # Assert deeply that results are the same and assign the results as one of them (either logits or distance)
            for idx_r in range(len(results_logits)):
                assert torch.allclose(results_logits[idx_r].boxes.xyxy, results_distance[idx_r].boxes.xyxy), f"Results are not the same for image {idx_r}"
                assert torch.allclose(results_logits[idx_r].boxes.cls, results_distance[idx_r].boxes.cls), f"Results are not the same for image {idx_r}"
                assert torch.allclose(results_logits[idx_r].boxes.conf, results_distance[idx_r].boxes.conf), f"Results are not the same for image {idx_r}"
            results = results_logits

            ### Fuse the results of the logits and distance methods ###
            ood_decision = self.fuse_ood_decisions(ood_decision_logits, ood_decision_distance)

            ### Añadir posibles cajas desconocidas a las predicciones ###
            if self.enhanced_unk_localization:
                if CUSTOM_HYP.unk.USE_XAI:
                    # TODO: Aqui pruebo lo de la explicabilidad, que tiene que sacar un mapa de valores por imagen, siendo cada mapa una imagen de 80x80
                    #   con los valores de la importancia de cada pixel, que luego se restaran al saliency map correspondiente, escalando los valores al rango
                    #   del saliency map
                    # Crear carpeta
                    # directory_name = f'{now}_{self.name}'
                    # imgs_folder_path = folder_path / directory_name
                    # imgs_folder_path.mkdir(exist_ok=True)
                    delattr(model.model, 'which_layers_to_extract')
                    delattr(model.model, 'extraction_mode')
                    # Heatmaps in shape (M, H, W), M being the batch size an in form of a tensor in cpu
                    # and in the range [0, 1]
                    if CUSTOM_HYP.unk.xai.XAI_METHOD == 'D-RISE':
                        from skimage.transform import downscale_local_mean
                        if p1 == 0.5:
                            save_name = f'./yolo_drise/heatmaps_{number_of_images_saved}_to_{count_of_images-1}.npy'
                        else:
                            save_name = f'./yolo_drise/heatmaps_{number_of_images_saved}_to_{count_of_images-1}_p1_{p1:.2f}.npy'
                        if not os.path.exists(save_name):
                            expl_heatmaps = expl_model(x=imgs, results=results, mode='object_detection')
                            # Save the heatmaps
                            np.save(save_name, expl_heatmaps.numpy())
                        else:
                            # Load the heatmaps and downscale them
                            print('***** LOADING HEATMAPS *****')
                            expl_heatmaps = torch.tensor(np.load(save_name), dtype=torch.float32)
                        processed_heatmaps = downscale_local_mean(expl_heatmaps.numpy(), (1, 8, 8))
                        processed_heatmaps = [torch.tensor(hm, dtype=torch.float32) for hm in processed_heatmaps]
                    else:
                        expl_heatmaps = expl_model(imgs, return_type='Tensor', show_image=False)
                        processed_heatmaps = expl_heatmaps
                        #processed_heatmaps = limit_heatmaps_to_bounding_boxes(expl_heatmaps, results)                 
                    configure_extra_output_of_the_model(model, self)
                    
                    # delattr(model.model, 'which_layers_to_extract')
                    # delattr(model.model, 'extraction_mode')
                    # expl_heatmaps = expl_model(imgs, return_type='Tensor', show_image=False)
                    # processed_heatmaps = limit_heatmaps_to_bounding_boxes(expl_heatmaps, results)                 
                    # configure_extra_output_of_the_model(model, self)
                else:
                    processed_heatmaps = None
                if CUSTOM_HYP.unk.RANK_BOXES:
                        possible_unk_bboxes, ood_decision_on_unknown, distances_per_image = self.compute_extra_possible_unkwnown_bboxes_and_decision(
                            results, data, ood_decision_of_results=ood_decision, explainalbility_heatmaps=processed_heatmaps,
                            folder_path=None, origin_of_idx=idx_of_batch*dataloader.batch_size
                        )
                else:
                    distances_per_image = None
                    possible_unk_bboxes, ood_decision_on_unknown = self.compute_extra_possible_unkwnown_bboxes_and_decision(
                        results, data, ood_decision_of_results=ood_decision, explainalbility_heatmaps=processed_heatmaps,
                        folder_path=None, origin_of_idx=idx_of_batch*dataloader.batch_size
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


### Method to configure internals of the model ###
    
def configure_extra_output_of_the_model(model: YOLO, ood_method: Type[OODMethod]):
        # Modify the model's attributes to output the desired extra_item
        # 1. Select the layers to extract depending on the OOD method from ultralytics/nn/tasks.py
        if ood_method.which_internal_activations in FTMAPS_RELATED_OPTIONS:
            model.model.which_layers_to_extract = "convolutional_layers"
        elif ood_method.which_internal_activations in LOGITS_RELATED_OPTIONS:
            model.model.which_layers_to_extract = "logits"
        elif ood_method.which_internal_activations == "none":
            model.model.which_layers_to_extract = "none"
        else:
            raise ValueError(f"The option {ood_method.which_internal_activations} is not valid.")
        # 2. Select the extraction mode for the ultralytics/yolo/v8/detect/predict.py
        model.model.extraction_mode = ood_method.which_internal_activations  # This attribute is created in the DetectionModel class
