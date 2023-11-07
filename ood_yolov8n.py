import time
import os
from pathlib import Path
from datetime import datetime
from typing import List
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.ops as t_ops
from torch.autograd import Variable
from torchvision.utils import draw_bounding_boxes
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment

import log
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results
from ood_utils import get_measures, arg_parser, OODMethod
from sos_dataset import SOS_BaseDataset
from data_utils import read_json, write_json, write_pickle, create_YOLO_dataset_and_dataloader, create_targets_dict


NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
STORAGE_PATH = Path('storage')
PRUEBAS_ROOT_PATH = Path('pruebas')

METHODS = {
    'msp': OODMethod(name='msp', per_class=True, per_stride=False, cluster_method='one', thresholds=None),
    'odin': OODMethod(name='odin', per_class=True, per_stride=False, cluster_method='one', thresholds=None),
}

############################################################

# Code partially copied from https://github.com/KingJamesSong/RankFeat

############################################################


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


def log_progress_of_batches(logger, idx_of_batch, number_of_batches):
    """
    Log the progress of batches every 50 batches or when 10, 25, 50 and 75% of 
    progress has been completed.
    """
    if idx_of_batch % 50 == 0:
        logger.info(f"{(idx_of_batch/number_of_batches) * 100:02.1f}%: Procesing batch {idx_of_batch} of {number_of_batches}")


def plot_results(class_names, results, valid_preds_only: bool, origin_of_idx: int, ood_decision=None, ood_method=None, image_format='pdf'):
    # ----------------------
    ### Codigo para dibujar las cajas predichas y los targets ###
    # ----------------------
    # Parametros para plot
    width = 2
    font = 'FreeMonoBold'
    font_size = 12

    if ood_decision:
        assert ood_method is not None, "If ood_decision exists, ood_method must be a string with the name of the OoD method"

    # Creamos la carpeta donde se guardaran las imagenes
    if ood_method:
        prueba_ahora_path = PRUEBAS_ROOT_PATH / (NOW + ood_method)
    else:
        prueba_ahora_path = PRUEBAS_ROOT_PATH / NOW
    prueba_ahora_path.mkdir(exist_ok=True)

    for img_idx, res in enumerate(results):
        # idx = torch.where(data['batch_idx'] == n_img)
        if valid_preds_only:
            valid_preds = np.array(res.valid_preds)
            bboxes = res.boxes.xyxy.cpu()[valid_preds]
            labels = res.boxes.cls.cpu()[valid_preds]
        else:
            bboxes = res.boxes.xyxy.cpu()
            labels = res.boxes.cls.cpu()

            # Este codigo es por si queremos que las cajas OoD se pinten con nombre OoD
            # for i, decision in enumerate(ood_decision[img_idx]):
            #     if decision == 0:
            #         labels[i] = 0
            # class_names[0] = 'OoD'

            # labels_for_plot = []
            # for i, lbl in enumerate(labels):
            #     if lbl == 0:
            #         labels_for_plot.append(f'{class_names[int(n.item())]} - {res.boxes.conf[i]:.2f}')
            #     else:    
            #         labels_for_plot.append(f'{class_names[int(n.item())]} - {res.boxes.conf[i]:.2f}') 
        
        # Si tenemos labels OOD, el plot lo hacemos para pintar 
        # de verde las cajas que son In-Distribution y de rojo las que son OoD
        if ood_decision:
            im = draw_bounding_boxes(
                res.orig_img[img_idx].cpu(),
                bboxes,
                width=width,
                font=font,
                font_size=font_size,
                labels=[f'{class_names[int(n.item())]} - {res.boxes.conf[i]:.2f}' for i, n in enumerate(labels)],
                colors=['red' if n == 0 else 'green' for n in ood_decision[img_idx]]
            )
        
        # Simplemente pintamos predictions
        else:
            im = draw_bounding_boxes(
                res.orig_img[img_idx].cpu(),
                bboxes,
                width=5,
                font=font,
                font_size=font_size,
                labels=[f'{class_names[int(n.item())]} - {res.boxes.conf[i]:.2f}' for i, n in enumerate(labels)]
            )

        plt.imshow(im.permute(1,2,0))
        plt.savefig(prueba_ahora_path / f'{(origin_of_idx + img_idx):03}.{image_format}', dpi=300)
        plt.close()

    # Code to plot an image with the targets
    # n_img = 4
    # idx = torch.where(data['batch_idx'] == n_img)
    # bboxes = t_ops.box_convert(data['bboxes'][idx], 'cxcywh', 'xyxy') * torch.Tensor([640, 640, 640, 640])
    # im = draw_bounding_boxes(
    #     imgs[n_img].cpu(), bboxes, width=5,
    #     font='FreeMono', font_size=8, labels=[model.names[int(n.item())] for n in data['cls'][idx]]
    # )
    # plt.imshow(im.permute(1,2,0))
    # plt.savefig('prueba.pdf')
    # plt.close()
    
    # Code to plot the predictions
    # from torchvision.utils import draw_bounding_boxes
    # n_img = 1
    # res = result[n_img]
    # # idx = torch.where(data['batch_idx'] == n_img)
    # bboxes = res.boxes.xyxy.cpu()
    # labels = res.boxes.cls.cpu()
    # im = draw_bounding_boxes(res.orig_img[n_img].cpu(), bboxes, width=5, font='FreeMonoBold', font_size=20, labels=[model.names[int(n.item())] for n in labels])
    # plt.imshow(im.permute(1,2,0))
    # plt.savefig('prueba.png')
    # plt.close()


def generate_predictions_with_ood_labeling(dataloader, model, device, logger, ood_thresholds, ood_method, temper=None, centroids=None):
    
    all_results = []

    conf = 0.15
    logger.warning(f"Using a confidence threshold of {conf} for tests")

    for idx_of_batch, data in enumerate(dataloader):
        
        ### Preparar imagenes y targets ###
        if isinstance(data, dict):
            imgs = data['img'].to(device)
            targets = create_targets_dict(data)
        else:
            imgs, targets = data
        
        ### Procesar imagenes en el modelo para obtener logits y las cajas ###
        results = model.predict(imgs, save=False, verbose=True, conf=conf)

        ### Comprobar si las cajas predichas son OoD ###
        ood_decision = []
        # Iteramos cada imagen
        for idx_img, res in enumerate(results):
            # TODO: Mejorar la forma en la que se guarda la decision de si es OoD o no.
            # TODO: Separar 
            ood_decision.append([])
            if centroids is not None:
                
                for stride_idx, (bbox_idx_in_one_stride, ftmaps) in enumerate(res.extra_item):

                    if len(bbox_idx_in_one_stride) > 0:  # Check if there are any predictions in this stride
                        # distances = pairwise_distances(centroids)

                        if ood_method == 'ftmaps':  # TODO: Hay que cambiar los métodoscl
                            
                            for idx, ftmap in enumerate(ftmaps):
                                bbox_idx = idx
                                cls_idx = int(res.boxes.cls[bbox_idx].cpu())
                                print('------------------------------')
                                print('idx_img:\t', idx_of_batch*dataloader.batch_size + idx_img)
                                print('bbox_idx:\t', bbox_idx)
                                print('cls:\t\t', cls_idx)
                                print('conf:\t\t', res.boxes.conf[bbox_idx])
                                # print('ftmap:\t\t',ftmap.shape)
                                # print('ftmap_reshape:\t', ftmap.cpu().numpy().reshape(1, -1).shape)

                                distance = pairwise_distances(centroids[cls_idx][stride_idx][None,:], ftmap.cpu().numpy().reshape(1, -1), metric='l1')
                                print('distance:\t', distance)
                                print('threshold:\t', ood_thresholds[cls_idx][stride_idx])

                                if distance < ood_thresholds[cls_idx][stride_idx]:
                                    ood_decision[idx_img].append(1)  # InD
                                else:
                                    ood_decision[idx_img].append(0)  # OOD
            

            # Logits methods
            else:
                # Iteramos cada bbox
                for idx_bbox in range(len(res.boxes.cls)):

                    # Extraemos la clase predicha y los logits
                    # TODO: Esto tiene que hacerse con las opciones de la clase
                    #   que vamos a crear para los métodos OOD
                    cls_idx = int(res.boxes.cls[idx_bbox].cpu())
                    logits = res.extra_item[idx_bbox][4:].cpu()

                    if ood_method == 'msp':
                        # Coger el cls_idx es como hacer el .max()
                        if logits[cls_idx] < ood_thresholds[cls_idx]:
                            ood_decision[idx_img].append(0)  # OOD
                        else:
                            ood_decision[idx_img].append(1)  # InD

                    elif ood_method == 'Energy':
                        energy_score = temper * torch.logsumexp(logits / temper, dim=0).item()
                        if energy_score < ood_thresholds[cls_idx]:
                            ood_decision[idx_img].append(0)  # OOD
                        else:
                            ood_decision[idx_img].append(1)  # InD
        
        plot_results(
            model.names,
            results,
            valid_preds_only=False,
            origin_of_idx=idx_of_batch*dataloader.batch_size,
            ood_decision=ood_decision,
            ood_method=ood_method,
            image_format='pdf'
        )
        
        if idx_of_batch > 10:
            quit()

    return all_results

def log_every_n_batches(logger, idx_of_batch: int, number_of_batches: int, log_every: int):
    if idx_of_batch % log_every == 0:
        logger.info(f"{(idx_of_batch/number_of_batches) * 100:02.1f}%: Procesing batch {idx_of_batch} of {number_of_batches}")

# -----------------------------------------------------------
# OOD Methods
# -----------------------------------------------------------

# Maximum Logit Score
def iterate_data_msp(data_loader, model: YOLO, device, logger):
    
    all_scores = [[] for _ in range(len(model.names))]
    number_of_batches = len(data_loader)
    
    # TODO: Quiza se pueda simplificar todo si no lo hacemos en batches, pero de momento vamos 
    # a intentarlo asi
    for idx_of_batch, data in enumerate(data_loader):
        
        log_every_n_batches(logger, idx_of_batch, number_of_batches, log_every=50)

        # ----------------------
        ### Preparar imagenes y targets ###
        # ----------------------
        if isinstance(data, dict):
            imgs = data['img'].to(device)
            targets = create_targets_dict(data)
        else:
            imgs, targets = data

        # ----------------------
        ### Procesar imagenes en el modelo para obtener logits y las cajas ###
        # ----------------------
        results = model.predict(imgs, save=False, verbose=False)

        # ----------------------
        ### Matchear cuales de las cajas han sido predichas correctamente ###
        # ----------------------
        # Matchea las cajas predichas a los targets y devuelve una lista
        # con los indices de las cajas predichas que han sido asignadas
        # dentro de results.valid_preds
        match_predicted_boxes_to_targets(results, targets, IOU_THRESHOLD)
        
        ### Acumular el Maximum Softmax Probability (MSP) de cada caja en la clase correspondiente ###
        # Recorremos cada imagen
        for res in results:
            # Recorremos cada caja predicha valida
            for valid_idx_one_bbox in res.valid_preds:
                # Extraemos el indice de la clase predicha para la caja actual
                cls_idx_one_bbox = int(res.boxes.cls[valid_idx_one_bbox].cpu())

                # Extraemos los logits de dicha caja
                logits_one_bbox = res.extra_item[valid_idx_one_bbox][4:].cpu().numpy()

                # Acumulamos en la clase correspondiente el Maximum Softmax Probability (MSP)
                # Hacemos el maximo se para obtener el MSP
                all_scores[cls_idx_one_bbox].append(logits_one_bbox.max())

        # # Plot results
        # plot_results(model, results)
        # quit()

    return all_scores

# ODIN Score
def iterate_data_odin(data_loader, model, device, logger, epsilon, temper):

    all_scores = [[] for _ in range(len(model.names))]
    number_of_batches = len(data_loader)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    confs = []
    for b, data in enumerate(data_loader):

        ### Preparar imagenes y targets ###
        if isinstance(data, dict):
            imgs = data['img'].to(device)
            targets = create_targets_dict(data)
        else:
            imgs, targets = data
        
        # 
        x = Variable(imgs, requires_grad=True)

        outputs =  model(x)

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        outputs = outputs / temper

        labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        tempInputs = torch.add(x.data, -epsilon, gradient)
        outputs = model(Variable(tempInputs))
        outputs = outputs / temper

        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        confs.extend(np.max(nnOutputs, axis=1))
        if b % 100 == 0:
            logger.info('{} batches processed'.format(b))

    return np.array(confs)

# Energy Score
def iterate_data_energy(data_loader, model, device, logger, temper):
    
    all_scores = [[] for _ in range(len(model.names))]
    number_of_batches = len(data_loader)

    for idx_of_batch, data in enumerate(data_loader):
        
        log_every_n_batches(logger, idx_of_batch, number_of_batches, log_every=50)
            
        ### Preparar imagenes y targets ###
        if isinstance(data, dict):
            imgs = data['img'].to(device)
            targets = create_targets_dict(data)
        else:
            imgs, targets = data

        ### Procesar imagenes en el modelo para obtener logits y las cajas ###
        results = model.predict(imgs, save=False, verbose=False)
        
        ### Matchear cuales de las cajas han sido predichas correctamente ###
        match_predicted_boxes_to_targets(results, targets, IOU_THRESHOLD)

        ### Acumular el Energy score de cada caja en la clase correspondiente ###
        # Recorremos cada imagen
        for res in results:
            # Recorremos cada caja predicha valida
            for valid_idx_one_bbox in res.valid_preds:
                # Extraemos el indice de la clase predicha para la caja actual
                cls_idx_one_bbox = int(res.boxes.cls[valid_idx_one_bbox].cpu())

                # Extraemos los logits de dicha caja
                logits_one_bbox = res.extra_item[valid_idx_one_bbox][4:].cpu()

                # Acumulamos en la clase correspondiente el Energy score
                all_scores[cls_idx_one_bbox].append(temper * torch.logsumexp(logits_one_bbox / temper, dim=0).item())

    return all_scores


# Feature Maps Score
def iterate_data_ftmaps(data_loader, model, device, logger):

    all_ftmaps_per_class_and_stride = [[[] for _ in range(3)] for _ in range(len(model.names))]
    number_of_batches = len(data_loader)

    for idx_of_batch, data in enumerate(data_loader):
        
        log_every_n_batches(logger, idx_of_batch, number_of_batches, log_every=50)
            
        ### Preparar imagenes y targets ###
        if isinstance(data, dict):
            imgs = data['img'].to(device)
            targets = create_targets_dict(data)
        else:
            imgs, targets = data

        ### Procesar imagenes en el modelo para obtener logits y las cajas ###
        results = model.predict(imgs, save=False, verbose=False)
        
        ### Matchear cuales de las cajas han sido predichas correctamente ###
        match_predicted_boxes_to_targets(results, targets, IOU_THRESHOLD)

        ### Extract the ftmaps of the selected boxes in their corresponding stride and class ###
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
                            all_ftmaps_per_class_and_stride[pred_cls][stride_idx].append(ftmaps[i].cpu().numpy())

    # Convert the list inside each class and stride to numpy arrays
    for idx_cls, ftmaps_one_cls in enumerate(all_ftmaps_per_class_and_stride):
        for idx_stride, ftmaps_one_cls_one_stride in enumerate(ftmaps_one_cls):
            if len(ftmaps_one_cls_one_stride) > 0:
                all_ftmaps_per_class_and_stride[idx_cls][idx_stride] = np.stack(ftmaps_one_cls_one_stride, axis=0)
            else:
                all_ftmaps_per_class_and_stride[idx_cls][idx_stride] = np.empty(0)

    return all_ftmaps_per_class_and_stride


def generate_ind_representation(
        agg_method: str,
        clustering_opt: str,
        ind_tensors: List[np.array] or List[List[np.array]],
        logger,
        per_class: bool,
        per_stride: bool,
        compute_covariance = False
    ) -> List[np.array] or List[List[np.array]]:
    """
    Generate the clusters for each class using the in-distribution tensors (usually feature maps).
    If per_stride, ind_tensors must be a list of lists, where each position is
        a list of tensors, one for each stride List[List[N, C, H, W]].
        Otherwise each position is just a tensor List[[N, C, H, W]].
    """
    CLUSTER_METHODS_AVAILABLE = ["KMeans", "DBSCAN", "AgglomerativeClustering"]
    
    if agg_method == 'mean':
        agg_method = np.mean
    elif agg_method == 'median':
        agg_method = np.median
    else:
        raise NameError(f"The agg_method argument must be one of the following: 'mean', 'median'. Current value: {agg_method}")

    
    if per_class:

        if per_stride:

            clusters_per_class_and_stride = [[[] for _ in range(3)] for _ in range(len(ind_tensors))]

            if clustering_opt == 'one':
                
                for cls_idx, ftmaps_one_cls in enumerate(ind_tensors):

                    print(f'Class {cls_idx:03} of {len(ind_tensors)}')
                    for stride_idx, ftmaps_one_cls_one_stride in enumerate(ftmaps_one_cls):
                        
                        if len(ftmaps_one_cls_one_stride) > 1:

                            ftmaps_one_cls_one_stride = ftmaps_one_cls_one_stride.reshape(ftmaps_one_cls_one_stride.shape[0], -1)
                            if compute_covariance:
                                clusters_per_class_and_stride = [
                                        agg_method(ftmaps_one_cls_one_stride, axis=0),
                                        np.cov(ftmaps_one_cls_one_stride, rowvar=False)  # rowvar to represent variables in columns
                                ]
                            else:
                                clusters_per_class_and_stride[cls_idx][stride_idx] = agg_method(ftmaps_one_cls_one_stride, axis=0)

                            if len(ftmaps_one_cls_one_stride) < 50:
                                print(f'WARNING: Class {cls_idx:03}, Stride {stride_idx} -> Only {len(ftmaps_one_cls_one_stride)} samples')

                        else:
                            print(f'WARNING: SKIPPING Class {cls_idx:03}, Stride {stride_idx} -> NO SAMPLES')
                            clusters_per_class_and_stride[cls_idx][stride_idx] = np.empty(0)

            elif clustering_opt in CLUSTER_METHODS_AVAILABLE:

                raise NotImplementedError("Not implemented yet")

            elif clustering_opt == 'all':
                raise NotImplementedError("As the amount of In-Distribution data is too big," \
                                        "ir would be intractable to treat each sample as a cluster")

            else:
                raise NameError(f"The clustering_opt must be one of the following: 'one', 'all', or one of{CLUSTER_METHODS_AVAILABLE}")
            
        else:
            raise NotImplementedError("Not implemented yet")
        
    else:
        raise NotImplementedError("Not implemented yet")

    return clusters_per_class_and_stride

def generate_ind_scores_for_distance_methods(method: str, ind_activations, ind_clusters, logger, options: SimpleNamespace):
    """
    Generate the scores of the in-distribution data using the in-distribution clusters.
    TODO: This function can be integrated with generating the thresholds as they can be computed on the
    fly inside the loop (optimization). For the moment, we will compute the thresholds separately for flexibilty.
    """
    # TODO: For the moment method is not being used
    if options.per_class:

        if options.per_stride:
                    
            ind_scores = [[[] for _ in range(3)] for _ in range(len(ind_activations))]

            for idx_cls, ind_activations_one_cls in enumerate(ind_activations):

                for idx_stride, ind_activations_one_cls_one_stride in enumerate(ind_activations_one_cls):

                    if len(ind_activations_one_cls_one_stride) > 0:
                        
                        # TODO: Aqui puedo introducir que el method sea una clase con funciones que computen
                        #   la distancia entre las activaciones

                        if len(ind_clusters[idx_cls][idx_stride]) > 0:
                            ind_scores[idx_cls][idx_stride].append(
                                pairwise_distances(ind_clusters[idx_cls][idx_stride][None,:],
                                                ind_activations_one_cls_one_stride.reshape(ind_activations_one_cls_one_stride.shape[0], -1),
                                                metric='l1')[0]
                            )
                        else:
                            print(f'WARNING: Class {idx_cls:03}, Stride {idx_stride} -> No clusters' \
                                   f'and {ind_activations_one_cls_one_stride.shape[0]} samples')

                    else:
                        print(f'WARNING: Class {idx_cls:03}, Stride {idx_stride} -> No samples')
    
    return ind_scores


def generate_ood_thresholds_for_distance_methods(ind_scores: list, tpr: float, options: SimpleNamespace, logger):
    """
    """
    if options.per_class:

        if options.per_stride:
                    
            thresholds = [[[] for _ in range(3)] for _ in range(len(ind_scores))]

            for idx_cls, ind_scores_one_cls in enumerate(ind_scores):
                for idx_stride, ind_scores_one_cls_one_stride in enumerate(ind_scores_one_cls):

                    if len(ind_scores_one_cls_one_stride) > 0:
                        # As the method is a distance method, we need to get the distance that makes the
                        # encompasses the tpr*100% of the data (more distance, more OOD)
                        thresholds[idx_cls][idx_stride] = np.percentile(ind_scores_one_cls_one_stride, tpr*100, method='lower')

                    else:
                        print(f'WARNING: Class {idx_cls:03}, Stride {idx_stride} -> No samples')
    
    return thresholds


def generate_ood_thresholds(ind_scores: list, tpr: float, logger, per_class: bool, dist_method=False) -> np.array:
    """
    Generate the thresholds for each class using the in-distribution scores.
    If per_class=True, in_scores must be a list of lists,
      where each list is the list of scores for each class.
    tpr must be in the range [0, 1]
    """
    if per_class:

        ood_thresholds = np.zeros(len(ind_scores))
        for idx, cl_scores in enumerate(ind_scores):
            if len(cl_scores) < 20:
                if len(cl_scores) > 10:
                    logger.warning(f"Class {idx} has {len(cl_scores)} samples. The threshold may not be accurate")
                    ood_thresholds[idx] = np.percentile(cl_scores, 100 - tpr*100, method='lower')
                else:
                    logger.warning(f"Class {idx} has less than 10 samples. The threshold is set to 0.2")
            else:
                ood_thresholds[idx] = np.percentile(cl_scores, 100 - tpr*100, method='lower')
    else:

        raise NotImplementedError("Not implemented yet")
    
    return ood_thresholds


def run_eval(model, device, in_loader, ood_loader, logger, args):
    logger.info("Running test...")
    logger.flush()

    if args.ood_method == 'MSP':
        
        # In scores
        if args.load_in_scores:
            logger.info("Loading in-distribution scores from disk...")
            in_scores = read_json(Path('pruebas/prueba_in_scores.json'))
        else:
            logger.info("Processing in-distribution data...")
            in_scores = iterate_data_msp(in_loader, model, device, logger)
            # Guardo los resultados en disco para no repetir el calculo
            in_scores_serializable = [[float(score) for score in cl] for cl in in_scores]
            write_json(in_scores_serializable, STORAGE_PATH / 'msp_in_scores.json')
            write_pickle(in_scores, STORAGE_PATH / 'msp_in_scores.pkl')
        
        ood_thresholds = generate_ood_thresholds(in_scores, tpr=0.95, per_class=True, logger=logger)

        # OoD scores
        logger.info("Processing out-of-distribution data...")
        ood_scores = generate_predictions_with_ood_labeling(ood_loader, model, device, logger, ood_thresholds, args.ood_method)

    elif args.ood_method == 'ODIN':
        raise NotImplementedError("Not implemented yet")
        # In scores
        if args.load_in_scores:
            logger.info("Loading in-distribution scores from disk...")
            in_scores = read_json(Path('pruebas/prueba_in_scores.json'))
        else:
            logger.info("Processing in-distribution data...")
            in_scores = iterate_data_odin(in_loader, model, device, logger, epsilon, temper)
            # Guardo los resultados en disco para no repetir el calculo
            in_scores_serializable = [[float(score) for score in cl] for cl in in_scores]
            write_json(in_scores_serializable, STORAGE_PATH / 'odin_in_scores.json')
            write_pickle(in_scores, STORAGE_PATH / 'odin_in_scores.pkl')
        
        ood_thresholds = generate_ood_thresholds(in_scores, tpr=0.95, per_class=True, logger=logger)

        # OoD scores
        logger.info("Processing out-of-distribution data...")
        ood_scores = generate_predictions_with_ood_labeling(ood_loader, model, device, logger, ood_thresholds)

        # logger.info("Processing in-distribution data...")
        # in_scores = iterate_data_odin(in_loader, model, args.epsilon_odin, args.temperature_odin, logger)
        # logger.info("Processing out-of-distribution data...")
        # ood_scores = iterate_data_odin(ood_loader, model, args.epsilon_odin, args.temperature_odin, logger)

    elif args.ood_method == 'Energy':

        # In scores
        if args.load_in_scores:
            logger.info("Loading in-distribution scores from disk...")
            in_scores = read_json(Path(STORAGE_PATH / 'energy_in_scores.json'))
        else:
            logger.info("Processing in-distribution data...")
            in_scores = iterate_data_energy(in_loader, model, device, logger, args.temperature_energy)
            # Guardo los resultados en disco para no repetir el calculo
            in_scores_serializable = [[float(score) for score in cl] for cl in in_scores]
            write_json(in_scores_serializable, STORAGE_PATH / 'energy_in_scores.json')
            write_pickle(in_scores, STORAGE_PATH / 'energy_in_scores.pkl')
        
        ood_thresholds = generate_ood_thresholds(in_scores, tpr=0.95, per_class=True, logger=logger)

        # OoD scores
        logger.info("Processing out-of-distribution data...")
        ood_scores = generate_predictions_with_ood_labeling(ood_loader, model, device, logger, ood_thresholds,
                                                            args.ood_method, temper=args.temperature_energy, )
        
    elif args.ood_method == 'ftmaps':

        # TODO: Abstraer todos los métodos a una clase con atributos de nombre, si es per_class, per_stride, etc.
        #   y que tenga funciones para testear un sample y así. De esta forma nos ahorramos el siguiente todo.
        # TODO: Tener un dict para cada metodo y que sea ese diccionario lo que se va pasando a las funciones
        #   y asi tener poca repeticion de codigo

        # El workflow es el siguiente para ftmaps methods:
        #   1. Generar los ftmaps de las imagenes de in-distribution. Cada metodo puede requerir pinchar en una capa
        #       distinta del modelo, por lo que hay que tenerlo en cuenta.
        #   2. Generar los clusters de cada clase usando los ftmaps de las imagenes de in-distribution.
        #       (Puede ser un solo cluster para toda la clase y el stride)
        #   3. Generar los scores de in-distribution usando los clusters de in-distribution. Hay que generarlos
        #       midiendo la distancia correspondiente al metodo que se use.
        #   4. Generar los thresholds fijando un TPR
        #   5. Generar los scores de out-of-distribution usando los thresholds de in-distribution

        # In scores
        if args.load_in_scores:
            logger.info("Loading in-distribution scores from disk...")
            in_activations = torch.load(STORAGE_PATH / 'ftmaps_in_scores.pt')
            logger.info("Finished loading!")

        else:
            logger.info("Processing in-distribution data...")
            in_activations = iterate_data_ftmaps(in_loader, model, device, logger)
            torch.save(in_activations, STORAGE_PATH / 'ftmaps_in_scores.pt', pickle_protocol=5)
            # # Save ftmaps to not repeat the calculations
            # ftmaps_dir = Path(STORAGE_PATH / f'ftmaps_ind_{model.ckpt["train_args"]["name"]}')
            # ftmaps_dir.mkdir(exist_ok=True)
            # for cl, ftmaps_one_cls in enumerate(in_scores):
            #     for stride_idx, ftmaps_one_cls_one_stride in enumerate(ftmaps_one_cls):
            #         if len(ftmaps_one_cls_one_stride) > 0:
            #             np.save(ftmaps_dir / f'cl-{cl:03}-{stride_idx}-stride.npy', ftmaps_one_cls_one_stride)

        ind_clusters = generate_ind_representation(
            agg_method='mean',
            clustering_opt='one',
            ind_tensors=in_activations,
            logger=logger,
            per_class=True,
            per_stride=True,
            compute_covariance=False, 
        )

        options = SimpleNamespace(per_class=True, per_stride=True, compute_covariance=False)

        in_scores = generate_ind_scores_for_distance_methods('l1', in_activations, ind_clusters, logger, options)

        ood_thresholds = generate_ood_thresholds_for_distance_methods(in_scores, tpr=0.95, options=options, logger=logger)
        
        # ood_thresholds = generate_ood_thresholds(in_scores, tpr=0.95, logger=logger, per_class=True, dist_method=True)

        # OoD scores
        logger.info("Processing out-of-distribution data...")
        ood_scores = generate_predictions_with_ood_labeling(
            ood_loader, model, device, logger, ood_thresholds, args.ood_method, centroids=ind_clusters
        )

        """
    elif args.ood_method == 'Mahalanobis':
        sample_mean, precision, lr_weights, lr_bias, magnitude = np.load(
            os.path.join(args.mahalanobis_param_path, 'results.npy'), allow_pickle=True)
        sample_mean = [s.cuda() for s in sample_mean]
        precision = [p.cuda() for p in precision]

        regressor = LogisticRegressionCV(cv=2).fit([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
                                                   [0, 0, 1, 1])

        regressor.coef_ = lr_weights
        regressor.intercept_ = lr_bias

        temp_x = torch.rand(2, 3, 480, 480)
        temp_x = Variable(temp_x).cuda()
        temp_list = model(x=temp_x, layer_index='all')[1]
        num_output = len(temp_list)

        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_mahalanobis(in_loader, model, num_classes, sample_mean, precision,
                                             num_output, magnitude, regressor, logger)
        logger.info("Processing out-of-distribution data...")
        ood_scores = iterate_data_mahalanobis(ood_loader, model, num_classes, sample_mean, precision,
                                              num_output, magnitude, regressor, logger)
    elif args.ood_method == 'GradNorm':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_gradnorm(in_loader, model, args.temperature_gradnorm, num_classes)
        logger.info("Processing out-of-distribution data...")
        ood_scores = iterate_data_gradnorm(ood_loader, model, args.temperature_gradnorm, num_classes)
    elif args.ood_method == 'RankFeat':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_rankfeat(in_loader, model, args.temperature_rankfeat)
        logger.info("Processing out-of-distribution data...")
        ood_scores = iterate_data_rankfeat(ood_loader, model, args.temperature_rankfeat)
    elif args.ood_method == 'React':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_react(in_loader, model, args.temperature_react)
        logger.info("Processing out-of-distribution data...")
        ood_scores = iterate_data_react(ood_loader, model, args.temperature_react)
        """
    else:
        raise ValueError("Unknown score type {}".format(args.ood_method))

    in_examples = in_scores.reshape((-1, 1))
    out_examples = ood_scores.reshape((-1, 1))

    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)

    logger.info('============Results for {}============'.format(args.ood_method))
    logger.info('AUROC: {}'.format(auroc))
    logger.info('AUPR (In): {}'.format(aupr_in))
    logger.info('AUPR (Out): {}'.format(aupr_out))
    logger.info('FPR95: {}'.format(fpr95))

    logger.flush()


def main(args):
    print('----------------------------')
    print('****************************')
    print('****************************')
    print('****************************')
    
    # Setup logger
    logger = log.setup_logger(args)

    # Make IoU threshold global
    global IOU_THRESHOLD 
    IOU_THRESHOLD = 0.5

    # TODO: This is for reproducibility 
    # torch.backends.cudnn.benchmark = True

    logger.warning('Changing following enviroment variables:')
    #os.environ['YOLO_VERBOSE'] = 'False'
    gpu_number = str(3)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_number
    logger.warning(f'CUDA_VISIBLE_DEVICES = {gpu_number}')

    # TODO: Unused till we implement GradNorm
    if args.ood_method == 'GradNorm':
        args.batch = 1

    # Load ID data and OOD data
    # TODO: Aqui tengo que meter algo que compruebe que el dataset esta como YAML file
    #if yaml_file_exists('coco.yaml'):
    
    ind_dataset, ind_dataloader = create_YOLO_dataset_and_dataloader(
        'coco.yaml',
        args,
        data_split='val',
    )
    # import matplotlib.pyplot as plt
    # a = ind_dataset[0]
    # plt.imshow(a['img'].permute(1,2,0))
    # plt.savefig('prueba.png')

    if True:
        ood_dataset, ood_dataloader = create_YOLO_dataset_and_dataloader(
            'VisDrone.yaml', 
            args=args,
            data_split='val',
        )

    if False:
        ood_dataset = SOS_BaseDataset(
            imgs_path='/home/tri110414/nfs_home/datasets/street_obstacle_sequences/raw_data/',
            ann_path='/home/tri110414/nfs_home/datasets/street_obstacle_sequences/val_annotations.json',
            imgsz=640
        )

        ood_dataloader = build_dataloader(
            ood_dataset,
            batch=args.batch_size,
            workers=args.num_workers,
            shuffle=False,
            rank=-1
        )

    # TODO: usar el argparser para elegir el modelo que queremos cargar
    model_to_load = 'yolov8n.pt'

    logger.info(f"Loading model {model_to_load} in {args.device}")

    logger.info(f"IoU threshold set to {IOU_THRESHOLD}")

    # Load YOLO model
    # TODO: add different YOLO models
    model = YOLO(model_to_load) 
    # state_dict = torch.load(args.model_path)
    # model.load_state_dict_custom(state_dict['model'])

    # TODO: Unused till we implement GradNorm
    # if args.ood_method != 'GradNorm':
    #     model = torch.nn.DataParallel(model)

    start_time = time.time()
    run_eval(model, args.device, ind_dataloader, ood_dataloader, logger, args)
    end_time = time.time()

    logger.info("Total running time: {}".format(end_time - start_time))


if __name__ == "__main__":
    parser = arg_parser()
    
    parser.add_argument('-d', '--device', default='cuda', type=str, help='use cpu or cuda')
    parser.add_argument('-b', '--batch_size', default=4, type=int, help='batch size to use')
    parser.add_argument('-n_w', '--num_workers', default=1, type=int, help='number of workers to use in dataloader')
    parser.add_argument('-l', '--load_in_scores', action='store_true', help='load in-distribution scores from disk')
    # parser.add_argument("--in_datadir", help="Path to the in-distribution data folder.")
    # parser.add_argument("--out_datadir", help="Path to the out-of-distribution data folder.")

    parser.add_argument('--ood_method',
                        choices=['MSP', 'ODIN', 'Energy', 'Mahalanobis', 'GradNorm','RankFeat','React', 'ftmaps'],
                        default='MSP')

    # arguments for ODIN
    parser.add_argument('--temperature_odin', default=1000, type=int,
                        help='temperature scaling for odin')
    parser.add_argument('--epsilon_odin', default=0.0, type=float,
                        help='perturbation magnitude for odin')

    # arguments for Energy
    parser.add_argument('--temperature_energy', default=1, type=int,
                        help='temperature scaling for energy')
    """
    # arguments for Mahalanobis
    parser.add_argument('--mahalanobis_param_path', default='checkpoints/finetune/tune_mahalanobis',
                        help='path to tuned mahalanobis parameters')

    # arguments for GradNorm
    parser.add_argument('--temperature_gradnorm', default=1, type=float,
                        help='temperature scaling for GradNorm')
    # arguments for React
    parser.add_argument('--temperature_react', default=1, type=float,
                        help='temperature scaling for React')
    # arguments for RankFeat
    parser.add_argument('--temperature_rankfeat', default=1, type=float,
                        help='temperature scaling for RankFeat')
    """
    print('******************************************')
    main(parser.parse_args())