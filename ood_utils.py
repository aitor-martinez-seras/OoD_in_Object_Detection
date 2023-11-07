from typing import List
from dataclasses import dataclass
from abc import abstractmethod

import argparse
import numpy as np
# import sklearn.metrics as sk
import torch
import torchvision.ops as t_ops
from scipy.optimize import linear_sum_assignment

from ultralytics.yolo.engine.results import Results


CLUSTERING_METHODS = ['DBSCAN', 'KMeans', 'GMM', 'HDBSCAN', 'OPTICS',
                      'SpectralClustering', 'AgglomerativeClustering']


@dataclass(slots=True)
class OODMethod:
    
    name: str
    distance_method: bool
    logits_or_ftmaps: str
    per_class: bool
    per_stride: bool
    cluster_method: str
    thresholds: List[float] or List[List[float]]
    iou_threshold_for_matching: float

    def __init__(self, name: str, per_class: bool, per_stride: bool, cluster_method: str, 
                 ):
        self.name = name
        self.per_class = per_class
        self.per_stride = per_stride
        self.cluster_method = self.check_clusters(cluster_method)

    def check_clusters(self, cluster_method: str) -> str:
        assert cluster_method in ['no','one', 'all'] + CLUSTERING_METHODS, f"Clusters must be either 'in', 'out' or {['one', 'all'] + CLUSTERING_METHODS}"

    @abstractmethod
    def compute_scores():
        pass

    def log_every_n_batches(self, logger, idx_of_batch: int, number_of_batches: int, log_every: int):
        if idx_of_batch % log_every == 0:
            logger.info(f"{(idx_of_batch/number_of_batches) * 100:02.1f}%: Procesing batch {idx_of_batch} of {number_of_batches}")
    
    @classmethod
    def create_targets_dict(data: dict) -> dict:
        """
        Funcion que crea un diccionario con los targets de cada imagen del batch.
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
        targets = dict(
            bboxes=[t_ops.box_convert(data['bboxes'][idx], 'cxcywh', 'xyxy') * relative_to_absolute_coordinates[img_idx] for img_idx, idx in enumerate(target_idx)],
            cls=[data['cls'][idx].view(-1) for idx in target_idx]    
        )
        return targets


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


    # TODO: Todavia me queda por decidir si esto es solo para cuando quiero extraer info interna para 
    #   modelar las In-Distribution o si tambien lo quiero para las Out-of-Distribution
    def iterate_data(self, data_loader, model, device, logger):
        """
        """        
        all_ftmaps_per_class_and_stride = [[[] for _ in range(3)] for _ in range(len(model.names))]
        number_of_batches = len(data_loader)

        for idx_of_batch, data in enumerate(data_loader):
            
            self.log_every_n_batches(logger, idx_of_batch, number_of_batches, log_every=50)
                
            ### Prepare images and targets to feed the model ###
            if isinstance(data, dict):
                imgs = data['img'].to(device)
                targets = self.create_targets_dict(data)
            else:
                imgs, targets = data

            ### Process the images to get the results (bboxes and clasification) and the extra info for OOD detection ###
            results = model.predict(imgs, save=False, verbose=False)
            
            ### Match the predicted boxes to the ground truth boxes ###
            self.match_predicted_boxes_to_targets(results, targets, self.iou_threshold_for_matching)

            ### Extract the internal information of the model depending on the OOD method ###
            self.extract_internal_information(results, all_ftmaps_per_class_and_stride)

        return all_ftmaps_per_class_and_stride
    
    @abstractmethod
    def extract_internal_information(self, results: Results, all_info: List[np.array] or List[List[np.array]]):
        """
        Function to be overriden by each method to extract the internal information of the model. In the logits
        methods, it will be the logits, and in the ftmaps methods, it will be the ftmaps.
        The extracted information will be stored in the list all_info
        """
        pass
    
    # TODO: Esta funcion puede valer para todos los metodos OOD
    def compute_ood_decision():
        pass

# Create classes for each method



############################################################

# Code copied from https://github.com/KingJamesSong/RankFeat/blob/main/utils/test_utils.py

############################################################


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--workers", type=int, default=0,
                        help="Number of background threads used to load data.")

    parser.add_argument("--logdir", default='logs',
                        help="Where to log test info (small).")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size.")
    parser.add_argument("--name", default='prueba',
                        help="Name of this run. Used for monitoring and checkpointing.")

    parser.add_argument("--model", default="YOLO", help="Which variant to use")
    parser.add_argument("--model_path", type=str, help="Path to the model you want to test")

    return parser


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