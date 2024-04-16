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
import torchvision.ops as t_ops
from scipy.optimize import linear_sum_assignment

from ultralytics import YOLO
from ultralytics.yolo.data.build import InfiniteDataLoader
from ultralytics.yolo.engine.results import Results
from visualization_utils import plot_results
from datasets_utils.owod.owod_evaluation_protocol import compute_metrics


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
        debugeando_en_modo = "localizacion"

        # TODO: Activado para ejecutar los plots OOD con targets
        if callable(getattr(self, "compute_ood_decision_with_ftmaps", None)):
            model.model.modo = 'all_ftmaps'

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
                        idx_of_img_per_box.append(torch.tensor([_idx]*len(one_img_bboxes), dtype=torch.int32))
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
                # Pintar los feature maps de algunas imagenes
                # Cogemos solo los mapas 80x80 de momento
                # Create folder to store images
                prueba_ahora_path = folder_path / f'{now}_{self.name}'
                prueba_ahora_path.mkdir(exist_ok=True)

                torch.set_printoptions(precision=2, threshold=10)
                np.set_printoptions(precision=2, threshold=10)

                for _img_idx, res in enumerate(results):
                    c += 1
                    if _img_idx == -1:
                        continue
                    else:
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
                        if True:
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
                            axs[0, 1].imshow(mean_ftmap, cmap='gray')
                            axs[0, 1].axis('off')
                            axs[0, 1].set_title('Mean')
                            axs[1, 0].imshow(std_ftmap, cmap='gray')
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
                            continue

                        # Create folder for the image using the batch index and image index
                        ftmaps_path = prueba_ahora_path / f'{number_of_images_saved + _img_idx}'
                        ftmaps_path.mkdir(exist_ok=True)
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
                        # # Save the feature maps
                        # for idx_ftmap in range(ftmaps.shape[0]):
                        #     # Reshape the feature map to original size
                        #     ftmap = ftmaps[idx_ftmap]

                        #     # Option 1: resize and plot
                        #     # ftmap = resize(ftmap, (imgs[_img_idx].shape[1], imgs[_img_idx].shape[2]), anti_aliasing=True)
                        #     # plt.imshow(ftmap, vmin=-2, vmax=2, cmap='bwr')

                        #     # Option 2: normalize to 0 - 1, resize and plot
                        #     ftmap = (ftmap - ftmap.min()) / (ftmap.max() - ftmap.min())
                        #     ftmap = resize(ftmap, (imgs[_img_idx].shape[1], imgs[_img_idx].shape[2]), anti_aliasing=True)
                        #     plt.imshow(ftmap, cmap='gray')

                        #     # Save close
                        #     plt.savefig(ftmaps_path / f'ftmap_{idx_ftmap}.pdf')
                        #     plt.close()
                        
                        # Make the mean of the feature maps
                        mean_ftmap = ftmaps.mean(axis=0)
                        mean_ftmap = (mean_ftmap - mean_ftmap.min()) / (mean_ftmap.max() - mean_ftmap.min())
                        # Add the padding
                        mean_ftmap = np.pad(mean_ftmap, ((y_padding, y_padding), (x_padding, x_padding)), 'constant', constant_values=(0, 0))
                        plt.imshow(mean_ftmap, cmap='gray')
                        plt.savefig(ftmaps_path / f'A_mean_ftmap.pdf')
                        plt.close()
                        mean_ftmap = resize(mean_ftmap, (imgs[_img_idx].shape[1], imgs[_img_idx].shape[2]), anti_aliasing=True)
                        plt.imshow(mean_ftmap, cmap='gray')
                        plt.savefig(ftmaps_path / f'A_mean_ftmap_reshaped.pdf')
                        plt.close()

                        # Make an image as the std of the feature maps
                        std_ftmap = ftmaps.std(axis=0)
                        std_ftmap = (std_ftmap - std_ftmap.min()) / (std_ftmap.max() - std_ftmap.min())
                        # Add the padding
                        std_ftmap = np.pad(std_ftmap, ((y_padding, y_padding), (x_padding, x_padding)), 'constant', constant_values=(0, 0))
                        plt.imshow(std_ftmap, cmap='gray')
                        plt.savefig(ftmaps_path / f'A_std_ftmap.pdf')
                        plt.close()
                        # Transform them to txt file and save it
                        np.savetxt(ftmaps_path / 'A_std_ftmap.txt', std_ftmap)
                        std_ftmap = resize(std_ftmap, (imgs[_img_idx].shape[1], imgs[_img_idx].shape[2]), anti_aliasing=True)
                        plt.imshow(std_ftmap, cmap='gray')
                        plt.savefig(ftmaps_path / f'A_std_ftmap_reshaped.pdf')
                        plt.close()

                        # Make an image as the max of the feature maps
                        max_ftmap = ftmaps.max(axis=0)
                        max_ftmap = (max_ftmap - max_ftmap.min()) / (max_ftmap.max() - max_ftmap.min())
                        # Add the padding
                        max_ftmap = np.pad(max_ftmap, ((y_padding, y_padding), (x_padding, x_padding)), 'constant', constant_values=(0, 0))
                        plt.imshow(max_ftmap, cmap='gray')
                        plt.savefig(ftmaps_path / f'A_max_ftmap.pdf')
                        plt.close()
                        max_ftmap = resize(max_ftmap, (imgs[_img_idx].shape[1], imgs[_img_idx].shape[2]), anti_aliasing=True)
                        plt.imshow(max_ftmap, cmap='gray')
                        plt.savefig(ftmaps_path / f'A_max_ftmap_reshaped.pdf')
                        plt.close()

                        # Make an image as the min of the feature maps -> GIVES NO INFO
                        min_ftmap = ftmaps.min(axis=0)
                        min_ftmap = (min_ftmap - min_ftmap.min()) / (min_ftmap.max() - min_ftmap.min())
                        # Add the padding
                        min_ftmap = np.pad(min_ftmap, ((y_padding, y_padding), (x_padding, x_padding)), 'constant', constant_values=(0, 0))
                        plt.imshow(min_ftmap, cmap='gray')
                        plt.savefig(ftmaps_path / f'A_min_ftmap.pdf')
                        plt.close()
                        min_ftmap = resize(min_ftmap, (imgs[_img_idx].shape[1], imgs[_img_idx].shape[2]), anti_aliasing=True)
                        plt.imshow(min_ftmap, cmap='gray')
                        plt.savefig(ftmaps_path / f'A_min_ftmap_reshaped.pdf')
                        plt.close()

                        # Make an image of the IQR of the feature maps usign scipy.stats.iqr
                        from scipy.stats import iqr
                        iqr_ftmap = iqr(ftmaps, axis=0)
                        iqr_ftmap = (iqr_ftmap - iqr_ftmap.min()) / (iqr_ftmap.max() - iqr_ftmap.min())
                        # Add the padding
                        iqr_ftmap = np.pad(iqr_ftmap, ((y_padding, y_padding), (x_padding, x_padding)), 'constant', constant_values=(0, 0))
                        plt.imshow(iqr_ftmap, cmap='gray')
                        plt.savefig(ftmaps_path / f'A_IQR_ftmap.pdf')
                        plt.close()

                        # Make and image of the MAD of the feature maps
                        mean_ftmaps = ftmaps.mean(axis=0)
                        mad_ftmap = np.mean(np.abs(ftmaps - mean_ftmaps), axis=0)
                        mad_ftmap = (mad_ftmap - mad_ftmap.min()) / (mad_ftmap.max() - mad_ftmap.min())
                        # Add the padding
                        mad_ftmap = np.pad(mad_ftmap, ((y_padding, y_padding), (x_padding, x_padding)), 'constant', constant_values=(0, 0))
                        plt.imshow(mad_ftmap, cmap='gray')
                        plt.savefig(ftmaps_path / f'A_MAD_ftmap.pdf')
                        plt.close()
                        
                        # Make an image of the sum of the topK values per pixel
                        #topK = 10
                        flattened_ftmaps = ftmaps.reshape(ftmaps.shape[0], -1).T
                        topK_folder_path = ftmaps_path / 'topK'
                        topK_folder_path.mkdir(exist_ok=True)
                        for topK in [1,2,3,4,5,8,10,12,15,20,25,30]:
                            topK_ftmap = np.sort(flattened_ftmaps, axis=1) # Sort the values (channels) of the pixels
                            topK_ftmap = topK_ftmap.reshape(ftmaps.shape[1], ftmaps.shape[2], -1)  # Reshape to the original shape
                            topK_ftmap = topK_ftmap[:,:, -topK:].sum(axis=2)  # Take the topK values of every pixel and sum them
                            topK_ftmap = (topK_ftmap - topK_ftmap.min()) / (topK_ftmap.max() - topK_ftmap.min())  # Normalize
                            # Add the padding
                            topK_ftmap = np.pad(topK_ftmap, ((y_padding, y_padding), (x_padding, x_padding)), 'constant', constant_values=(0, 0))
                            #topK_ftmap = topK_ftmap.reshape(ftmaps.shape[1], ftmaps.shape[2])
                            plt.imshow(topK_ftmap, cmap='gray')
                            plt.savefig(topK_folder_path / f'A_top{topK}_ftmap.pdf')
                            plt.close()
                            topK_ftmap = resize(topK_ftmap, (imgs[_img_idx].shape[1], imgs[_img_idx].shape[2]), anti_aliasing=True)
                            plt.imshow(topK_ftmap, cmap='gray')
                            plt.savefig(topK_folder_path / f'A_top{topK}_ftmap_reshaped.pdf')
                            plt.close()

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
                        plt.imshow(binary_map, cmap='gray')
                        plt.savefig(ftmaps_path / f'A_binary_map_otsu.pdf')
                        plt.close()

                        # Adaptive Mean Thresholding
                        adaptive_mean = cv2.adaptiveThreshold(saliency_map_8bit, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
                        plt.imshow(adaptive_mean, cmap='gray')
                        plt.savefig(ftmaps_path / f'A_binary_map_adaptive_mean.pdf')
                        plt.close()

                        # Adaptive Gaussian Thresholding
                        adaptive_gaussian = cv2.adaptiveThreshold(saliency_map_8bit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                        plt.imshow(adaptive_gaussian, cmap='gray')
                        plt.savefig(ftmaps_path / f'A_binary_map_adaptive_gaussian.pdf')
                        plt.close()

                        # Triangle thresholding
                        _, triangle_threshold = cv2.threshold(saliency_map_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
                        plt.imshow(triangle_threshold, cmap='gray')
                        plt.savefig(ftmaps_path / f'A_binary_map_triangle.pdf')
                        plt.close()

                        from skimage.filters import threshold_multiotsu
                        # Applying Multi-Otsu threshold for the values in image
                        thresholds = threshold_multiotsu(saliency_map_8bit, classes=3)
                        multi_otsu_result = np.digitize(saliency_map_8bit, bins=thresholds)
                        plt.imshow(multi_otsu_result, cmap='gray')
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
        known_classes_tensor = torch.tensor(known_classes, dtype=torch.float32)
        for idx_of_batch, data in enumerate(dataloader):

            if idx_of_batch % 50 == 0 or idx_of_batch == number_of_batches - 1:
                logger.info(f"{(idx_of_batch/number_of_batches) * 100:02.1f}%: Procesing batch {idx_of_batch+1} of {number_of_batches}") 

            ### Preparar imagenes y targets ###
            imgs, targets = self.prepare_data_for_model(data, device)
            
            ### Procesar imagenes en el modelo para obtener logits y las cajas ###
            results = model.predict(imgs, save=False, verbose=False, conf=self.min_conf_threshold)

            ### Comprobar si las cajas predichas son OoD ###
            ood_decision = self.compute_ood_decision_on_results(results, logger)

            # Cada prediccion va a ser un diccionario con las siguientes claves:
            #   'img_idx': int -> Indice de la imagen
            #   'img_name': str -> Nombre del archivo de la imagen
            #   'bboxes': List[torch.Tensor] -> Lista de tensores con las cajas predichas
            #   'cls': List[torch.Tensor] -> Lista de tensores con las clases predichas
            #   'conf': List[torch.Tensor] -> Lista de tensores con las confianzas de las predicciones (en yolov8 es cls)
            #   'ood_decision': List[int] -> Lista de enteros con la decision de si la caja es OoD o no
            for img_idx, res in enumerate(results):
                #for idx_bbox in range(len(res.boxes.cls)):
                # Parse the ood elements as the unknown class (80)
                ood_decision_one_image = torch.tensor(ood_decision[img_idx], dtype=torch.float32)
                unknown_mask = ood_decision_one_image == 0
                bboxes_cls = torch.where(unknown_mask, torch.tensor(80, dtype=torch.float32), res.boxes.cls.cpu())
                all_preds.append({
                    'img_idx': number_of_images_processed + img_idx,
                    'img_name': Path(data['im_file'][img_idx]).stem,
                    'bboxes': res.boxes.xyxy,
                    'cls': bboxes_cls,
                    'conf': res.boxes.conf,
                    'ood_decision': torch.tensor(ood_decision[img_idx], dtype=torch.float32)
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

    def compute_ood_decision_with_ftmaps(self, activations: Union[List[np.array], List[List[np.array]]], bboxes: Dict[str, List], logger: Logger) -> List[List[List[int]]]:
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


class CosineDistanceOneClusterPerStride(DistanceMethod):
    
        # name: str, agg_method: str, per_class: bool, per_stride: bool, cluster_method: str, cluster_optimization_metric: str
        def __init__(self, agg_method, **kwargs):
            name = 'CosineDistancePerStride'
            per_class = True
            per_stride = True
            cluster_method = 'one'
            cluster_optimization_metric = 'silhouette'
            super().__init__(name, agg_method, per_class, per_stride, cluster_method, cluster_optimization_metric, **kwargs)
        
        def compute_distance(self, cluster: np.array, activations: np.array) -> List[float]:

            distances = pairwise_distances(
                cluster,
                activations,
                metric='cosine'
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