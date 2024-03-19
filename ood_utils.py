from typing import List, Tuple, Callable, Type, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
from logging import Logger
import json 

import numpy as np
# import sklearn.metrics as sk
from sklearn.metrics import pairwise_distances
import torch
import torchvision.ops as t_ops
from scipy.optimize import linear_sum_assignment

from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results
from visualization_utils import plot_results


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
        which_internal_activations: str -> Where to extract internal activations from. It can be 'logits' or 'ftmaps'
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

    def __init__(self, name: str, distance_method: bool, per_class: bool, per_stride: bool, iou_threshold_for_matching: float,
                 min_conf_threshold: float, which_internal_activations: str):
        self.name = name
        self.distance_method = distance_method
        self.per_class = per_class
        self.per_stride = per_stride
        self.iou_threshold_for_matching = iou_threshold_for_matching
        self.min_conf_threshold = min_conf_threshold
        self.thresholds = None
        self.which_internal_activations = which_internal_activations

    @abstractmethod
    def extract_internal_activations(self, results: Results, all_activations: Union[List[float], List[List[np.array]]]):
        """
        Function to be overriden by each method to extract the internal activations of the model. In the logits
        methods, it will be the logits, and in the ftmaps methods, it will be the ftmaps.
        The extracted activations will be stored in the list all_activations
        """
        pass

    @abstractmethod
    def format_internal_activations(self, all_activations: Union[List[float], List[List[np.array]]]):
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
    def create_targets_dict(data: dict) -> dict:
        """
        Funcion que crea un diccionario con los targets de cada imagen del batch con el siguiente formato:
            targets = {
                'bboxes': [list of bboxes for each image],
                'cls': [list of classes for each image],
            }
        Las bboxes se llevan de coordenadas relativas a absolutas y se convierten a formato xyxy
        La funcion es necesaria porque los targets vienen en una sola dimension, 
            por lo que hay que hacer el matcheo de a que imagen pertenece cada bbox.
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
        Funcion que matchea las cajas predichas con las cajas Ground Truth (GT) que mejor se ajustan.
        El matching se devuelve en la variable results.valid_preds, que es una lista de indices de las cajas
        cuyas predicciones han matcheado con una caja GT con un IoU mayor que el threshold.
        """
        # TODO: Optimizar 
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
            self.extract_internal_activations(results, all_internal_activations)

        ### Final formatting of the internal activations ###
        self.format_internal_activations(all_internal_activations)

        return all_internal_activations

    def prepare_data_for_model(self, data, device) -> Tuple[torch.Tensor, dict]:
        """
        Funcion que prepara los datos para poder meterlos en el modelo.
        """
        if isinstance(data, dict):
            imgs = data['img'].to(device)
            targets = self.create_targets_dict(data)
        else:
            imgs, targets = data
        return imgs, targets

    def iterate_data_to_plot_with_ood_labels(self, model, dataloader, device, logger, folder_path: Path, now: str):
        
        # # Obtain the bbox format from the last transform of the dataset
        # if hasattr(dataloader.dataset.transforms.transforms[-1], "bbox_format"):
        #     bbox_format = dataloader.dataset.transforms.transforms[-1].bbox_format
        # else:
        #     bbox_format=dataloader.dataset.labels[0]['bbox_format']

        logger.warning(f"Using a confidence threshold of {self.min_conf_threshold} for tests")
        count_of_images = 0
        number_of_images_saved = 0
        for idx_of_batch, data in enumerate(dataloader):
            count_of_images += len(data['im_file'])
            if count_of_images < 8000:
                continue
            ### Preparar imagenes y targets ###
            imgs, targets = self.prepare_data_for_model(data, device)
            
            ### Procesar imagenes en el modelo para obtener logits y las cajas ###
            results = model.predict(imgs, save=False, verbose=True, conf=self.min_conf_threshold)

            ### Comprobar si las cajas predichas son OoD ###
            ood_decision = self.compute_ood_decision_on_results(results, logger)
            
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
            number_of_images_saved += len(data['im_file'])
            # TODO: De momento no queremos plotear todo, solo unos pocos batches
            if number_of_images_saved > 200:
                quit()
            # if idx_of_batch > 10:
            #     quit()
                
    def iterate_data_to_compute_metrics(self, model, dataloader, device, logger, folder_path: Path, now: str):

        logger.warning(f"Using a confidence threshold of {self.min_conf_threshold} for tests")
        count_of_images = 0
        number_of_images_processed = 0
        all_preds = []
        all_targets = []
        for idx_of_batch, data in enumerate(dataloader):

            ### Preparar imagenes y targets ###
            imgs, targets = self.prepare_data_for_model(data, device)
            
            ### Procesar imagenes en el modelo para obtener logits y las cajas ###
            results = model.predict(imgs, save=False, verbose=True, conf=self.min_conf_threshold)

            ### Comprobar si las cajas predichas son OoD ###
            ood_decision = self.compute_ood_decision_on_results(results, logger)

            ### Calcular las metricas de las cajas predichas ###
        
            number_of_images_processed += len(data['im_file'])
    
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

#################################################################################
# Create classes for each method. Methods will inherit from OODMethod,
#   will override the abstract methods and also any other function that is needed.
#################################################################################

### Superclasses for methods using logits of the model ###
class LogitsMethod(OODMethod):
    
    def __init__(self, name: str, per_class: bool, per_stride: bool, iou_threshold_for_matching: float, min_conf_threshold: float):
        distance_method = False
        which_internal_activations = 'logits'
        super().__init__(name, distance_method, per_class, per_stride, iou_threshold_for_matching, min_conf_threshold, which_internal_activations)

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
    
    def extract_internal_activations(self, results: Results, all_activations: List[float]) -> List[float]:
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

    def format_internal_activations(self, all_activations: Union[List[float], List[List[np.array]]]):
        """
        Format the internal activations of the model. In this case, the activations are already well formatted.
        """
        pass


class MSP(LogitsMethod):

    def __init__(self, **kwargs):
        name = 'MSP'
        super().__init__(name, **kwargs)
    
    def compute_score_one_bbox(self, logits: torch.Tensor, cls_idx: int) -> float:
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


class ODIN(LogitsMethod):
    pass


### Superclasses for methods using feature maps of the model ###
class DistanceMethod(OODMethod):
    
    agg_method: Callable
    cluster_method: str
    cluster_optimization_metric: str
    available_cluster_methods: List[str]
    available_cluster_optimization_metrics: List[str]
    clusters: Union[List[np.array], List[List[np.array]]]

    # name: str, distance_method: bool, per_class: bool, per_stride: bool, iou_threshold_for_matching: float, min_conf_threshold: float
    def __init__(self, name: str, agg_method: str, per_class: bool, per_stride: bool, cluster_method: str, cluster_optimization_metric: str, **kwargs):
        distance_method = True  # Always True for distance methods
        which_internal_activations = 'ftmaps'  # This could be changed in subclasses
        super().__init__(name, distance_method, per_class, per_stride, which_internal_activations=which_internal_activations,**kwargs)
        self.available_cluster_methods = ['one','all','DBSCAN', 'KMeans', 'GMM', 'HDBSCAN', 'OPTICS', 'SpectralClustering', 'AgglomerativeClustering']
        self.available_cluster_optimization_metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
        self.cluster_method = self.check_cluster_method_selected(cluster_method)
        self.cluster_optimization_metric = self.check_cluster_optimization_metric_selected(cluster_optimization_metric)
        if agg_method == 'mean':
            self.agg_method = np.mean
        elif agg_method == 'median':
            self.agg_method = np.median
        else:
            raise NameError(f"The agg_method argument must be one of the following: 'mean', 'median'. Current value: {agg_method}")
        
    # TODO: Quiza estas formulas acaben devolviendo un Callable con el propio método que implementen
    def check_cluster_method_selected(self, cluster_method: str) -> str:
        assert cluster_method in self.available_cluster_methods, f"cluster_method must be one of {self.available_cluster_methods}, but got {cluster_method}"
        return cluster_method

    def check_cluster_optimization_metric_selected(self, cluster_optimization_metric: str) -> str:
        assert cluster_optimization_metric in self.available_cluster_optimization_metrics, f"cluster_method must be one" \
          f"of {self.available_cluster_optimization_metrics}, but got {cluster_optimization_metric}"
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

    def extract_internal_activations(self, results: Results, all_activations: List[List[np.array]]):
        """
        Extract the ftmaps of the selected boxes in their corresponding stride and class.
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
        # Loop each image fo the batch
        for res in results:
            cls_idx_one_pred = res.boxes.cls.cpu()
            # Recorremos cada stride y de ahí nos quedamos con las cajas que hayan sido marcadas como validas
            # Loop each stride and get only the ftmaps of the boxes that are valid predictions
            for stride_idx, (bbox_idx_in_one_stride, ftmaps) in enumerate(res.extra_item):
                if len(bbox_idx_in_one_stride) > 0:  # Check if there are any valid predictions in this stride
                    for i, bbox_idx in enumerate(bbox_idx_in_one_stride):
                        bbox_idx = bbox_idx.item()
                        if bbox_idx in res.valid_preds:
                            pred_cls = int(cls_idx_one_pred[bbox_idx].item())
                            all_activations[pred_cls][stride_idx].append(ftmaps[i].cpu().numpy())

    def format_internal_activations(self, all_activations: List[List[List[np.array]]]):
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
   

    def compute_scores_from_activations(self, activations: Union[List[np.array], List[List[np.array]]], logger: Logger):
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
    def compute_scores_one_cluster_per_class_and_stride(self, activations: List[List[np.array]], scores: List[List[np.array]], logger):
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
        ood_decision = []  
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

        return ood_decision

    def generate_clusters(self, ind_tensors: Union[List[np.array], List[List[np.array]]], logger: Logger) -> Union[List[np.array], List[List[np.array]]]:
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
    
    def generate_one_cluster_per_class_and_stride(self, ind_tensors: List[List[np.array]], clusters_per_class_and_stride: List[List[np.array]], logger):
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
        def __init__(self, agg_method, **kwargs):
            name = 'L1DistancePerStride'
            per_class = True
            per_stride = True
            cluster_method = 'one'
            cluster_optimization_metric = 'silhouette'
            super().__init__(name, agg_method, per_class, per_stride, cluster_method, cluster_optimization_metric, **kwargs)
        
        def compute_distance(self, cluster: np.array, activations: np.array) -> List[float]:

            distances = pairwise_distances(
                cluster,
                activations,
                metric='l1'
                )

            return distances[0]
        

class L2DistanceOneClusterPerStride(DistanceMethod):
    
        # name: str, agg_method: str, per_class: bool, per_stride: bool, cluster_method: str, cluster_optimization_metric: str
        def __init__(self, agg_method, **kwargs):
            name = 'L2DistancePerStride'
            per_class = True
            per_stride = True
            cluster_method = 'one'
            cluster_optimization_metric = 'silhouette'
            super().__init__(name, agg_method, per_class, per_stride, cluster_method, cluster_optimization_metric, **kwargs)
        
        def compute_distance(self, cluster: np.array, activations: np.array) -> List[float]:

            distances = pairwise_distances(
                cluster,
                activations,
                metric='l2'
                )

            return distances[0]
        

class GAPL2DistanceOneClusterPerStride(DistanceMethod):
    
        # name: str, agg_method: str, per_class: bool, per_stride: bool, cluster_method: str, cluster_optimization_metric: str
        def __init__(self, agg_method, **kwargs):
            name = 'GAP_L2DistancePerStride'
            per_class = True
            per_stride = True
            cluster_method = 'one'
            cluster_optimization_metric = 'silhouette'
            super().__init__(name, agg_method, per_class, per_stride, cluster_method, cluster_optimization_metric, **kwargs)
        
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
    def __init__(self, agg_method, **kwargs):
        name = 'ActivationsExtractor'
        per_class = True
        per_stride = True
        cluster_method = 'one'
        cluster_optimization_metric = 'silhouette'
        super().__init__(name, agg_method, per_class, per_stride, cluster_method, cluster_optimization_metric, **kwargs)

    def compute_distance(self, cluster: np.array, activations: np.array) -> List[float]:
        raise NotImplementedError("Not implemented yet")
    
    def activations_transformation(self, activations: np.array) -> np.array:
        """
        Transform the activations to the shape needed to compute the distance.
        By default, it flattens the activations leaving the batch dimension as the first dimension.
        """
        raise NotImplementedError("Not implemented yet")
        return np.mean(activations, axis=(2,3))  # Already reshapes to [N, features]
    
    def iterate_data_to_extract_ind_activations_and_create_its_annotations(self, data_loader, model, device, split: str):
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
            self.extract_internal_activations(results, all_internal_activations)

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
    def __init__(self, agg_method, **kwargs):
        name = 'FeaturemapExtractor'
        per_class = True
        per_stride = True
        cluster_method = 'one'
        cluster_optimization_metric = 'silhouette'
        super().__init__(name, agg_method, per_class, per_stride, cluster_method, cluster_optimization_metric, **kwargs)

    def compute_distance(self, cluster: np.array, activations: np.array) -> List[float]:
        raise NotImplementedError("Not implemented yet")
    
    def activations_transformation(self, activations: np.array) -> np.array:
        """
        Transform the activations to the shape needed to compute the distance.
        By default, it flattens the activations leaving the batch dimension as the first dimension.
        """
        raise NotImplementedError("Not implemented yet")
        return np.mean(activations, axis=(2,3))  # Already reshapes to [N, features]
    
    def iterate_data_to_extract_ind_activations_and_create_its_annotations(self, data_loader, model, device, split: str):
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
            self.extract_internal_activations(results, all_internal_activations)

            # TODO: Ir recolectando los targets para poder hacer el match con las predicciones.
            #   Recolectar haciendo una lista de listas de listas similar, solo que en vez de un array
            #   por posicion va a ser un dict por cada posicion con la anotacion correspondiente

        ### Final formatting of the internal activations ###
        self.format_internal_activations(all_internal_activations)

        # TODO: Tras recolectar, hay que asignar las anotaciones a las predicciones y hacer el match pero
        #   manteniendo el orden de los videos

        return all_internal_activations

    

### Other methods ###

def configure_extra_output_of_the_model(model: YOLO, ood_method: Type[OODMethod]):
        
        # TODO: Tenemos que definir que un atributo de los ood methods define de donde sacar
        #   el extra_item. De momento nos limitamos a usar el modo "logits" y "ftmaps"
        if isinstance(ood_method, LogitsMethod):
            modo = 'logits'
        elif isinstance(ood_method, DistanceMethod):
            modo = 'conv'
        else:
            raise NotImplementedError("Not implemented yet")
        
        # Modify the model's attributes to output the desired extra_item
        model.model.modo = modo  # This attribute is created in the DetectionModel class

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