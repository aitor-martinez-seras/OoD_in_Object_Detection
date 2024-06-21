# -*- coding: utf-8 -*-
##############################################################################

# Below code is modified from https://github.com/scuwyh2000/RandBox/blob/main/randbox/pascal_voc_evaluation.py

##############################################################################

# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import numpy as np
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from typing import List, Dict
from logging import Logger
from collections import OrderedDict, defaultdict
from functools import lru_cache
from pathlib import Path
import json

import matplotlib.pyplot as plt
import torch
from torch.distributions.weibull import Weibull
from torch.distributions.transforms import AffineTransform
from torch.distributions.transformed_distribution import TransformedDistribution

from constants import UNKNOWN_CLASS_INDEX


OWOD_FOLDER_PATH = Path(__file__).parent


def compute_avg_precision_at_many_recall_level_for_unk(precisions, recalls):
    precs = {}
    for r in range(1, 10):
        r = r/10
        p = compute_avg_precision_at_a_recall_level_for_unk(precisions, recalls, recall_level=r)
        precs[r] = p
    return precs


def compute_avg_precision_at_a_recall_level_for_unk(precisions, recalls, recall_level=0.5):
    precs = {}
    for iou, recall in recalls.items():
        prec = []
        for cls_id, rec in enumerate(recall):
            if cls_id == 20 and len(rec)>0:  # 20 as the number of classes is 21 and the number of known classes is 20
                p = precisions[iou][cls_id][min(range(len(rec)), key=lambda i: abs(rec[i] - recall_level))]
                prec.append(p)
        #print(prec)
        if len(prec) > 0:
            precs[iou] = np.mean(prec)
        else:
            precs[iou] = 0
    return precs


def compute_WI_at_many_recall_level(recalls, tp_plus_fp_cs, fp_os, known_classes):
    wi_at_recall = {}
    for r in range(1, 10):
        r = r/10
        try:
            wi = compute_WI_at_a_recall_level(recalls, tp_plus_fp_cs, fp_os, recall_level=r, known_classes=known_classes)
        except TypeError as e:  # Added this to avoid errors when there are no predictions for a class
            print(e)
            wi = 100
        wi_at_recall[r] = wi
    return wi_at_recall


def compute_WI_at_a_recall_level(recalls, tp_plus_fp_cs, fp_os, recall_level=0.5, known_classes=None):
    wi_at_iou = {}
    num_seen_classes = len(known_classes)
    for iou, recall in recalls.items():
        tp_plus_fps = []
        fps = []
        for cls_id, rec in enumerate(recall):
            if cls_id in range(num_seen_classes) and len(rec) > 0:
                index = min(range(len(rec)), key=lambda i: abs(rec[i] - recall_level))
                tp_plus_fp = tp_plus_fp_cs[iou][cls_id][index]
                tp_plus_fps.append(tp_plus_fp)
                fp = fp_os[iou][cls_id][index]
                fps.append(fp)
        if len(tp_plus_fps) > 0:
            wi_at_iou[iou] = np.mean(fps) / np.mean(tp_plus_fps)
        else:
            wi_at_iou[iou] = 0
    return wi_at_iou


def compute_metrics(all_predictions: List[Dict], all_targets: List[Dict], class_names: List[str], known_classes: List[int], logger: Logger) -> Dict[str, float]:
    """
    Returns:
        dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
    """
    # predictions, targets, logger, class_names, known_classes, prev_intro_cls, curr_intro_cls, 

    with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
        res_file_template = os.path.join(dirname, "{}.txt")

        aps = defaultdict(list)  # iou -> ap per class
        recs = defaultdict(list)
        precs = defaultdict(list)
        all_recs = defaultdict(list)
        all_precs = defaultdict(list)
        unk_det_as_knowns = defaultdict(list)
        num_unks = defaultdict(list)
        tp_plus_fp_cs = defaultdict(list)
        fp_os = defaultdict(list)

        # For each class, compute some metrics
        for cls_id, cls_name in enumerate(class_names[:len(known_classes)] + ['unknown']):
            # Get the predictions with this class as predicted class
            one_class_preds = {
                "image_names": [],
                "confs": [],
                "bboxes": [],
            }
            if cls_name == 'unknown':
                cls_id = UNKNOWN_CLASS_INDEX
            for one_img_preds in all_predictions:
                current_cls_preds_mask = one_img_preds['cls'] == cls_id
                one_class_preds["image_names"].extend([one_img_preds['img_name'] for _ in range(current_cls_preds_mask.sum())])
                one_class_preds["confs"].extend(one_img_preds['conf'][current_cls_preds_mask].tolist())
                one_class_preds["bboxes"].extend(one_img_preds['bboxes'][current_cls_preds_mask].tolist())
            one_class_preds["confs"] = np.array(one_class_preds["confs"], dtype=np.float32)
            one_class_preds["bboxes"] = np.array(one_class_preds["bboxes"], dtype=np.float32)
            
            # for thresh in range(50, 100, 5):
            thresh = 50  # IoU threshold

            # In case there are no predictions for current class, just append empty arrays and the num unks
            if one_class_preds["confs"].shape[0] == 0:
                logger.info(f"No predictions for class {cls_name}")
                empty_array = np.empty(0)
                # aps[thresh].append(ap * 100)
                # unk_det_as_knowns[thresh].append(0)
                # # Compute num unks
                # num_unk = 0
                # for target in all_targets:
                #     current_class_gt_mask = target['cls'] == UNKNOWN_CLASS_INDEX  # 80 es el indice de unknown
                #     R = target["cls"][current_class_gt_mask]
                #     difficult = np.array([False for x in R]).astype(bool)  # All are non difficult
                #     num_unk = num_unk + sum(~difficult)  # Number of non-difficult GTs, always same as len(R)
                # num_unks[thresh].append(num_unk)
                #all_precs[thresh].append(empty_array)
                all_recs[thresh].append(empty_array)
                tp_plus_fp_cs[thresh].append(empty_array)
                fp_os[thresh].append(empty_array)
                continue
            
            # In case there are predictions, compute the metrics
            rec, prec, ap, unk_det_as_known, num_unk, tp_plus_fp_closed_set, fp_open_set = voc_eval(
                current_class_predictions=one_class_preds,
                all_targets=all_targets,
                imagesetfile=(OWOD_FOLDER_PATH / "tasks/all_task_test.txt").as_posix(),  # Se suele coger all_task_test.txt
                classname=cls_name,
                ovthresh=thresh / 100.0,
                use_07_metric=False,
                known_classes=known_classes,
                class_names=class_names,
                logger=logger
            )
            aps[thresh].append(ap * 100)
            unk_det_as_knowns[thresh].append(unk_det_as_known)
            num_unks[thresh].append(num_unk)
            all_precs[thresh].append(prec)
            all_recs[thresh].append(rec)
            tp_plus_fp_cs[thresh].append(tp_plus_fp_closed_set)
            fp_os[thresh].append(fp_open_set)
            try:
                recs[thresh].append(rec[-1] * 100)
                precs[thresh].append(prec[-1] * 100)
            except:
                recs[thresh].append(0)
                precs[thresh].append(0)
            logger.info(f"Class: {cls_name}, AP: {ap * 100:.2f}, Recall: {rec[-1] * 100:.2f}, Precision: {prec[-1] * 100:.2f}")       
    
    # Compute metrics for UNK as in UnkSniffer
    detections_unksniffer = {}
    there_are_unk_preds = False
    for cls_id, cls_name in enumerate(class_names[:len(known_classes)] + ['unknown']):
        one_class_preds ={}
        if cls_name == 'unknown':
            cls_id = UNKNOWN_CLASS_INDEX
        for one_img_preds in all_predictions:
            current_cls_preds_mask = one_img_preds['cls'] == cls_id
            if cls_id == UNKNOWN_CLASS_INDEX and current_cls_preds_mask.sum() > 0:
                there_are_unk_preds = True
            # Obtain an array of shape [N,5], where N is the number of predictions for this class
            # and 5 is the format [x1, y1, x2, y2, conf]
            one_class_preds[one_img_preds['img_name']] = np.concatenate([one_img_preds['bboxes'][current_cls_preds_mask], one_img_preds['conf'][current_cls_preds_mask][:, None]], axis=1)
        detections_unksniffer[cls_id] = one_class_preds
    # annotations =  dict[image_file] = numpy.array([[x1,y1,x2,y2, cl_id], [...],...])
    annotations_unksniffer = {}
    for one_img_targets in all_targets:
        annotations_unksniffer[one_img_targets['img_name']] = np.concatenate([one_img_targets['bboxes'], one_img_targets['cls'][:, None]], axis=1)

    # Compute metrics for Known classes
    aps_unksniffer = defaultdict(list)  # iou -> ap per class
    _all_recs = defaultdict(list)
    _all_precs = defaultdict(list)
    _unk_det_as_knowns = defaultdict(list)
    _tp_plus_fp_cs = defaultdict(list)
    _fp_os = defaultdict(list)
    for cls_id, cls_name in enumerate(class_names[:len(known_classes)]):
        _rec, _prec, _ap, _num_det_as_unk, _num_unks_gt, _tp_plus_fp_closed_set, _fp_open_set = voc_eval_unksniffer_WI_file(
                    detections=detections_unksniffer,
                    annotations=annotations_unksniffer,
                    classname=cls_id,
                )
        aps_unksniffer[thresh].append(_ap * 100)
        _all_precs[thresh].append(_prec)
        _all_recs[thresh].append(_rec)
        _unk_det_as_knowns[thresh].append(_num_det_as_unk)
        _tp_plus_fp_cs[thresh].append(_tp_plus_fp_closed_set)
        _fp_os[thresh].append(_fp_open_set)
        #logger.info(f"Class: {cls_name}, AP: {_ap * 100:.2f}, Recall: {_rec * 100:.2f}, Precision: {_prec * 100:.2f}")
    known_ap50_unksniffer = np.mean(aps_unksniffer[50])
    
    # Compute metrics for UNK
    if there_are_unk_preds:
        recall_unksniff, precision_unksniff, ap_unksniff, _, _, state, det_image_files = voc_evaluate_as_unksniffer(
                detections=detections_unksniffer,
                annotations=annotations_unksniffer,
                cid=UNKNOWN_CLASS_INDEX,
            )
        
    else:
        recall_unksniff, precision_unksniff, ap_unksniff, f1_unksniff = 0, 0, 0, 0
    f1_unksniff = 2 * (precision_unksniff * recall_unksniff) / (precision_unksniff + recall_unksniff) if precision_unksniff + recall_unksniff > 0 else 0
    print("---------------")
    logger.info(f"Class: UNK from UnkSniffer\nU-AP:\t{ap_unksniff * 100:.3f}\nU-F1':\t{f1_unksniff * 100:.3f}\nU-PRE:\t{precision_unksniff * 100:.3f}\nU-REC:\t{recall_unksniff * 100:.3f}")
    print("---------------")
    # Check if we are in coco_ood, where ONLY the unknown class is present in the targets.
    # If is the case, we need to return the metrics for the unknown class ONLY
    # Otherwise, we compute WI and A-OSE and return all metrics
    cls_diff_from_unk_found = False
    for t in all_targets:
        if torch.where(t["cls"] == UNKNOWN_CLASS_INDEX, 0, 1).sum() > 0:
            cls_diff_from_unk_found = True
            break
    if not cls_diff_from_unk_found:
        return {
            'U-AP': ap_unksniff,
            'U-F1': f1_unksniff,
            'U-PRE': precision_unksniff,
            'U-REC': recall_unksniff,
        }
    
    # WI is calculated for all_test?? Or only for WI split?
    wi = compute_WI_at_many_recall_level(all_recs, tp_plus_fp_cs, fp_os, known_classes=known_classes)
    logger.info('----------------- Calculations as per Towards Open World Object Detection paper -----------------')
    logger.info('Wilderness Impact: ' + str(wi))
    logger.info('Wilderness Impact Recall 0.8: ' + str(wi[0.8]))

    avg_precision_unk = compute_avg_precision_at_many_recall_level_for_unk(all_precs, all_recs)
    logger.info('avg_precision: ' + str(avg_precision_unk))
    ret = OrderedDict()
    mAP = {iou: np.mean(x) for iou, x in aps.items()}
    ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50]}
    
    total_num_unk_det_as_known = {iou: np.sum(x) for iou, x in unk_det_as_knowns.items()}
    total_num_unk = num_unks[50][0]
    logger.info('Absolute OSE (total_num_unk_det_as_known): ' + str(total_num_unk_det_as_known))
    logger.info('total_num_unk ' + str(total_num_unk))
    # Extra logging of class-wise APs
    avg_precs = list(np.mean([x for _, x in aps.items()], axis=0))
    logger.info(class_names)
    # logger.info("AP__: " + str(['%.1f' % x for x in avg_precs]))
    logger.info("AP50: " + str(['%.1f' % x for x in aps[50]]))
    logger.info("Precisions50: " + str(['%.1f' % x for x in precs[50]]))
    logger.info("Recall50: " + str(['%.1f' % x for x in recs[50]]))
    # logger.info("AP75: " + str(['%.1f' % x for x in aps[75]]))
    # logger.info("\nKnown AP__: " + str(np.mean(avg_precs[:self.prev_intro_cls + self.curr_intro_cls])))
    logger.info("Known AP50: " + str(np.mean(aps[50][:-1])))
    logger.info("Known Precisions50: " + str(np.mean(precs[50][:-1])))
    logger.info("Known Recall50: " + str(np.mean(recs[50][:-1])))
    logger.info("------------------------------------------------------------------------------------------------\n")

    # Metrics from Towards Open World Object Detection paper
    known_ap50 = np.mean(aps[50][:-1])
    unknown_ap50 = aps[50][-1]
    unknown_f1 = 2 * (precs[50][-1] * recs[50][-1]) / (precs[50][-1] + recs[50][-1])
    unknown_precision = precs[50][-1]
    unknown_recall = recs[50][-1]
    wilderness_impact_recall_08 = wi[0.8][50]
    total_num_unk_det_as_known = total_num_unk_det_as_known[50]
    # Reporting metrics from UnkSniffer code except for A-OSE and WI
    results_dict = {
        'mAP': known_ap50_unksniffer/100,
        'U-AP': ap_unksniff,
        'U-F1': f1_unksniff,
        'U-PRE': precision_unksniff,
        'U-REC': recall_unksniff,
        'A-OSE': total_num_unk_det_as_known,
        'WI-08': wilderness_impact_recall_08,
    }
    logger.info('Summary using UnkSniffer code for eval (except A-OSE and WI):')
    logger.info('-----------------')
    logger.info('Known mAP50 [%]: ' + str(known_ap50_unksniffer))
    logger.info('Unknown AP50 [%]: ' + str(ap_unksniff))
    logger.info('Unknown F1 score [%]: ' + str(f1_unksniff))
    logger.info('Unknown Precision [%]: ' + str(precision_unksniff))
    logger.info('Unknown Recall [%]: ' + str(recall_unksniff))
    logger.info('A-OSE: ' + str(total_num_unk_det_as_known))
    logger.info('Wilderness Impact Recall 0.8 [%]: ' + str(wilderness_impact_recall_08))
    logger.info('-----------------')

    return results_dict


##############################################################################
#
# Below code is modified from
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

"""Python implementation of the PASCAL VOC devkit's AP evaluation code."""


@lru_cache(maxsize=None)
def parse_rec(filename, known_classes):
    """Parse a PASCAL VOC xml file."""
    VOC_CLASS_NAMES_COCOFIED = [
        "airplane", "dining table", "motorcycle",
        "potted plant", "couch", "tv"
    ]
    BASE_VOC_CLASS_NAMES = [
        "aeroplane", "diningtable", "motorbike",
        "pottedplant", "sofa", "tvmonitor"
    ]
    try:
        with PathManager.open(filename) as f:
            tree = ET.parse(f)
    except:
        logger = logging.getLogger(__name__)
        logger.info('Not able to load: ' + filename + '. Continuing without aboarting...')
        return None

    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        cls_name = obj.find("name").text
        if cls_name in BASE_VOC_CLASS_NAMES:
            cls_name = VOC_CLASS_NAMES_COCOFIED[BASE_VOC_CLASS_NAMES.index(cls_name)]
        if cls_name not in known_classes:
            cls_name = 'unknown'
        obj_struct["name"] = cls_name
        # obj_struct["pose"] = obj.find("pose").text
        # obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(current_class_predictions: Dict[str, List], all_targets: List[Dict], imagesetfile, classname, ovthresh=0.5, use_07_metric=False,
             known_classes: List[int] = None, class_names: List[str] = None, logger: Logger = None):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,  # 'all_tasks_test.txt'
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    known_classes: List of known classes idx
    class_names: List of class names
    logger: Logger object
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name

    # Classname to idx
    class_mapping = {class_name: idx if class_name != 'unknown' else UNKNOWN_CLASS_INDEX for idx, class_name in enumerate(class_names)}
    class_idx = class_mapping[classname]  # Tengo que llevar hasta aqui el mapping de las clases

    # MANTENGO
    # first load gt
    # read list of images (all_tasks_test.txt)
    with open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # ELIMINO
    # load annots. Nosotros directamente las tenemos ya en memoria y lo que hacemos es asegurar que toda imagen tiene un target
    # recs = {}
    # for imagename in imagenames:
    #     rec = parse_rec(annopath.format(imagename), tuple(known_classes))
    #     if rec is not None:  # Es None para los XML a los que no se puede acceder por lo que sea. Asumimos que siempre podemos acceder a todos.
    #         recs[imagename] = rec
    #         imagenames_filtered.append(imagename)
    # imagenames = imagenames_filtered
    
    # NOTE: recs es un diccionario con las anotaciones de cada imagen.
    #   Las keys son los filenames y dentro de cada key el value es una list con cada anotacion. 
    #   A su vez, cada anotacion es un diccionario con las keys:
    #       - name: nombre de la clase
    #       - difficult: si es dificil o no
    #       - bbox: [xmin, ymin, xmax, ymax]

    # Assert every imagename of the set is present in the targets
    if False:  # This check is only for OWOD... As we also use COCO Mixed we cannot do this check as it is now
        imagenames_set = set(imagenames)
        target_imagenames_set = set([x['img_name'] for x in all_targets])
        assert imagenames_set == target_imagenames_set, 'Some images are missing in the targets'

    ### Coger el GT de la clase actual para TODAS las imagenes y formatear al estilo de ellos ###
    # ELIMINO y CAMBIO
    # extract gt objects for this class
    # class_recs = {}
    # npos = 0
    # for imagename in imagenames:
    #     R = [obj for obj in recs[imagename] if obj["name"] == classname]  # De cada imagen, cogemos los objetos de la clase classname
    #     bbox = np.array([x["bbox"] for x in R])
    #     difficult = np.array([x["difficult"] for x in R]).astype(np.bool_)
    #     # difficult = np.array([False for x in R]).astype(np.bool_)  # treat all "difficult" as GT
    #     det = [False] * len(R)
    #     npos = npos + sum(~difficult)
    #     class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}
    class_recs = {}
    npos = 0
    for target in all_targets:
        #if target['img_name'] in imagenames:
        imagename = target['img_name']
        current_class_gt_mask = target['cls'] == class_idx
        R = target["cls"][current_class_gt_mask]
        bbox = target["bboxes"][current_class_gt_mask].numpy()
        difficult = np.array([False for x in R]).astype(bool)  # All are non difficult
        det = [False] * len(R)
        npos = npos + sum(~difficult)  # Number of non-difficult GTs, always same as len(R)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    ### Coger las predicciones solo de la clase predicha (muchas imagenes se quedaran fuera) y formatear al estilo de ellos ###
    # ELIMINO y CAMBIO
    # read dets
    # detfile = detpath.format(classname)
    # with open(detfile, "r") as f:
    #     lines = f.readlines()
    # splitlines = [x.strip().split(" ") for x in lines]
    # image_ids = [x[0] for x in splitlines]
    # confidence = np.array([float(x[1]) for x in splitlines])
    # BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)
    
    # TODO: De aqui tiene que salir una variable con los img names de las predicciones,
    #   otra con las confidences y otra con las bboxes, todas en el mismo orden y misma longitud. 
    #   la longitud es el numero de predicciones (cajas) de la clase actual
    image_names = current_class_predictions.get("image_names")
    confidence = current_class_predictions.get("confs")
    BB = current_class_predictions.get('bboxes')

    # sort by confidence
    # sorted_ind = np.argsort(-confidence)
    # BB = BB[sorted_ind, :]
    # image_ids = [image_ids[x] for x in sorted_ind]

    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    if len(sorted_ind) == 0:
        raise ValueError('No predictions for class ' + classname)
        return [0], [0], 0, 0, 0, 0, 0
    image_names = [image_names[x] for x in sorted_ind]  # CAMBIO

    # go down dets and mark TPs and FPs
    #nd = len(image_ids)
    nd = len(image_names)  # CAMBIO
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    # if 'unknown' not in classname:
    #     return tp, fp, 0

    ## For each detection, check the GT it overlaps with ##
    # Assign a 1 to tp if it overlaps with a GT not already detected and it is not difficult
    # Assign a 1 to fp if it does not overlap with any GT or it overlaps with a GT already detected
    for d in range(nd):
        # Yo en vez de coger el file id y transformarlo a name cojo directamente el name
        #R = class_recs[str(mapping[int(image_ids[d])])]  # CAMBIO . El mapping lo que hace es mapear el id de la imagen a su nombre para asi acceder a su GT
        R = class_recs[image_names[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)  # recall = TP / (TP + FN). npos is the number of instances of the wanted class (TP + FN)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    # plot_pr_curve(prec, rec, classname+'.png')
    ap = voc_ap(rec, prec, use_07_metric)

    # print('tp: ' + str(tp[-1]))
    # print('fp: ' + str(fp[-1]))
    # print('tp: ')
    # print(tp)
    # print('fp: ')
    # print(fp)

    '''
    Computing Absolute Open-Set Error (A-OSE) and Wilderness Impact (WI)
                                    ===========    
    Absolute OSE = # of unknown objects classified as known objects of class 'classname'
    WI = FP_openset / (TP_closed_set + FP_closed_set)
    '''
    #logger = logging.getLogger(__name__)

    # ELIMINO y CAMBIO
    # Finding GT of unknown objects
    # unknown_class_recs = {}
    # n_unk = 0
    # for imagename in imagenames:
    #     R = [obj for obj in recs[imagename] if obj["name"] == 'unknown']
    #     bbox = np.array([x["bbox"] for x in R])
    #     difficult = np.array([x["difficult"] for x in R]).astype(np.bool_)
    #     det = [False] * len(R)
    #     n_unk = n_unk + sum(~difficult)
    #     unknown_class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}
    # TODO: Aqui tengo que asegurarme de haber designado como unknown aquellas classes que no son conocidas
    unknown_class_recs = {}
    n_unk = 0
    for target in all_targets:
        #if target['img_name'] in imagenames:
        imagename = target['img_name']
        current_class_gt_mask = target['cls'] == UNKNOWN_CLASS_INDEX  # 80 es el indice de unknown
        R = target["cls"][current_class_gt_mask]
        bbox = target["bboxes"][current_class_gt_mask].numpy()
        difficult = np.array([False for x in R]).astype(bool)  # All are non difficult
        det = [False] * len(R)
        n_unk = n_unk + sum(~difficult)  # Number of non-difficult GTs, always same as len(R)
        unknown_class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    if classname == 'unknown':
        return rec, prec, ap, 0, n_unk, None, None

    # Go down each detection and see if it has an overlap with an unknown object.
    # If so, it is an unknown object that was classified as known.
    is_unk = np.zeros(nd)
    for d in range(nd):
        #R = unknown_class_recs[str(mapping[int(image_ids[d])])]  # CAMBIO
        R = unknown_class_recs[image_names[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            is_unk[d] = 1.0

    is_unk_sum = np.sum(is_unk)
    # OSE = is_unk / n_unk
    # logger.info('Number of unknowns detected knowns (for class '+ classname + ') is ' + str(is_unk))
    # logger.info("Num of unknown instances: " + str(n_unk))
    # logger.info('OSE: ' + str(OSE))

    tp_plus_fp_closed_set = tp+fp
    fp_open_set = np.cumsum(is_unk)

    return rec, prec, ap, is_unk_sum, n_unk, tp_plus_fp_closed_set, fp_open_set


def plot_pr_curve(precision, recall, filename, base_path='/home/fk1/workspace/OWOD/output/plots/'):
    fig, ax = plt.subplots()
    ax.step(recall, precision, color='r', alpha=0.99, where='post')
    ax.fill_between(recall, precision, alpha=0.2, color='b', step='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.savefig(base_path + filename)


# Below code is obtained and modified from https://github.com/Went-Liang/UnSniffer/blob/main/detection/evaluator/voc_eval_offical.py#L51 

def voc_evaluate_as_unksniffer(detections, annotations, cid, ovthresh=0.5, use_07_metric=True):
    """
    Top level function that does the PASCAL VOC evaluation.
    :param detections: Bounding box detections dictionary, keyed on class id (cid) and image_file,
                       dict[cid][image_file] = numpy.array([[x1,y1,x2,y2,score], [...],...])
    :param annotations: Ground truth annotations, keyed on image_file,
                       dict[image_file] = numpy.array([[x1,y1,x2,y2,score], [...],...])
    :param cid: Class ID (0 is typically reserved for background, but this function does not care about the value)
    :param ovthresh: Intersection over union overlap threshold, above which detection is considered as correct,
                       if it matches to a ground truth bounding box along with its class label (cid)
    :param use_07_metric: Whether to use VOC 2007 metric
    :return: recall, precision, ap (average precision)
    """
    # detections {81: [np, np, np...]}, np = numpy.array([[x1,y1,x2,y2,score], [...],...])
    # annotations [np, np, np], np = numpy.array([[x1,y1,x2,y2,class_id], [...],...])
    # cid = 81
    import copy

    # extract ground truth objects from the annotations for this class
    class_gt_bboxes = {}
    npos = 0  # number of ground truth bboxes having label cid
    # annotations keyed on image file names or paths or anything that is unique for each image
    #for image_name in annotations:
    for image_name in annotations.keys():
        # for each image list of objects: [[x1,y1, x2,y2, cid], [], ...]
        R = [obj[:4] for obj in annotations[image_name] if int(obj[-1]) == cid]
        bbox = np.array(R)
        # difficult is not stored: take it as 0/false
        difficult = np.array([0] * len(R)).astype(np.bool_)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_gt_bboxes[image_name] = {'bbox': bbox, 'difficult': difficult, 'det': det}

    # detections' image file names/paths
    det_image_files = []
    confidences = []
    det_bboxes = []
    # detections should be keyed on class_id (cid)
    class_dict = detections[cid]
    #for image_file in class_dict:
    for image_file in class_dict.keys():
        dets = class_dict[image_file]
        for k in range(dets.shape[0]):
            det_image_files.append(image_file)
            det_bboxes.append(dets[k, 0:4])
            confidences.append(dets[k, -1])
    det_bboxes = np.array(det_bboxes)
    confidences = np.array(confidences)

    # number of detections
    num_dets = len(det_image_files)
    tp = np.zeros(num_dets)
    fp = np.zeros(num_dets)

    if det_bboxes.shape[0] == 0:
        return 0., 0., 0., 0., 0., None, None

    # sort detections by confidence
    sorted_ind = np.argsort(-confidences)
    det_bboxes = det_bboxes[sorted_ind, :]
    det_image_files = [det_image_files[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    for d in range(num_dets):
        R = class_gt_bboxes[det_image_files[d]]
        bb = det_bboxes[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            ## compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            # IoU
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    state = [copy.deepcopy(tp), copy.deepcopy(fp)]
    # compute precision recall
    stp = sum(tp)
    recall = stp / npos
    precision = stp / (stp + sum(fp))

    # compute average precision
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return recall, precision, ap, rec, prec, state, det_image_files


# Below code is obtained and modified from https://github.com/Went-Liang/UnSniffer/blob/main/detection/evaluator/WI.py#L63

def voc_eval_unksniffer_WI_file(detections, annotations, classname, ovthresh=0.5, use_07_metric=True, known_classes=None):
    # detections {81: [np, np, np...]}, np = np.array([[x1,y1,x2,y2,score], [...],...])
    # annotations [np, np, np], np = np.array([[x1,y1,x2,y2,class_id], [...],...])
    # classname = 81
    # extract ground truth objects from the annotations for this class
    import copy
    class_gt_bboxes = {}
    npos = 0  # number of ground truth bboxes having label classname
    # annotations keyed on image file names or paths or anything that is unique for each image
    for image_name in annotations:
        # for each image list of objects: [[x1,y1, x2,y2, classname], [], ...]
        R = [obj[:4] for obj in annotations[image_name] if int(obj[-1]) == classname]
        bbox = np.array(R)
        # difficult is not stored: take it as 0/false
        difficult = np.array([0] * len(R)).astype(np.bool_)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_gt_bboxes[image_name] = {'bbox': bbox, 'difficult': difficult, 'det': det}

    # detections' image file names/paths
    det_image_files = []
    confidences = []
    det_bboxes = []
    # detections should be keyed on class_id (classname)
    class_dict = detections[classname]
    for image_file in class_dict:
        dets = class_dict[image_file]
        for k in range(dets.shape[0]):
            det_image_files.append(image_file)
            det_bboxes.append(dets[k, 0:4])
            confidences.append(dets[k, -1])
    det_bboxes = np.array(det_bboxes)
    confidences = np.array(confidences)

    # number of detections
    num_dets = len(det_image_files)
    tp = np.zeros(num_dets)
    fp = np.zeros(num_dets)

    if det_bboxes.shape[0] == 0:
        unknown_class_recs = {}
        n_unk = 0
        if classname == UNKNOWN_CLASS_INDEX:
            for image_name in annotations:
                R = [obj[:4] for obj in annotations[image_name] if int(obj[-1]) == UNKNOWN_CLASS_INDEX]
                bbox = np.array(R)
                difficult = np.array([0] * len(R)).astype(np.bool_)
                det = [False] * len(R)
                n_unk = n_unk + sum(~difficult)
                unknown_class_recs[image_name] = {"bbox": bbox, "difficult": difficult, "det": det}
        return 0., 0., 0., 0, n_unk, None, None

    # sort detections by confidence
    sorted_ind = np.argsort(-confidences)
    det_bboxes = det_bboxes[sorted_ind, :]
    det_image_files = [det_image_files[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    for d in range(num_dets):
        if det_image_files[d] not in class_gt_bboxes:
            continue
        R = class_gt_bboxes[det_image_files[d]]
        bb = det_bboxes[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            ## compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            # IoU
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    state = [copy.deepcopy(tp), copy.deepcopy(fp)]
    # compute precision recall
    stp = sum(tp)
    recall = stp / npos
    precision = stp / (stp + sum(fp))

    # compute average precision
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    '''
    Computing Absolute Open-Set Error (A-OSE) and Wilderness Impact (WI)
                                    ===========    
    Absolute OSE = # of unknown objects classified as known objects of class 'classname'
    WI = FP_openset / (TP_closed_set + FP_closed_set)
    '''

    # Finding GT of unknown objects
    unknown_class_recs = {}
    n_unk = 0
    for image_name in annotations:
        R = [obj[:4] for obj in annotations[image_name] if int(obj[-1]) == UNKNOWN_CLASS_INDEX]
        bbox = np.array(R)
        difficult = np.array([0] * len(R)).astype(np.bool_)
        det = [False] * len(R)
        n_unk = n_unk + sum(~difficult)
        unknown_class_recs[image_name] = {"bbox": bbox, "difficult": difficult, "det": det}

    if classname == UNKNOWN_CLASS_INDEX:
        return rec, prec, ap, 0, n_unk, None, None

    # Go down each detection and see if it has an overlap with an unknown object.
    # If so, it is an unknown object that was classified as known.
    is_unk = np.zeros(num_dets)

    for d in range(num_dets):
        R = unknown_class_recs[det_image_files[d]]
        bb = det_bboxes[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            is_unk[d] = 1.0

    is_unk_sum = np.sum(is_unk)
    # OSE = is_unk / n_unk
    # logger.info('Number of unknowns detected knowns (for class '+ classname + ') is ' + str(is_unk))
    # logger.info("Num of unknown instances: " + str(n_unk))
    # logger.info('OSE: ' + str(OSE))

    tp_plus_fp_closed_set = tp+fp
    fp_open_set = np.cumsum(is_unk)
    # print(fp_open_set)
    # print(len(fp_open_set))

    return rec, prec, ap, is_unk_sum, n_unk, tp_plus_fp_closed_set, fp_open_set

# _aps = defaultdict(list)  # iou -> ap per class
# _all_recs = defaultdict(list)
# _all_precs = defaultdict(list)
# _tp_plus_fp_cs = defaultdict(list)
# _fp_os = defaultdict(list)
# for cls_id, cls_name in enumerate(class_names[:len(known_classes)] + ['unknown']):
#     if cls_name == 'unknown':
#         cls_id = UNKNOWN_CLASS_INDEX
#     _rec, _prec, _ap, _d, _e, _tp_plus_fp_closed_set, _fp_open_set = voc_eval_unksniffer_WI_file(
#                 detections=detections_unksniffer,
#                 annotations=annotations_unksniffer,
#                 classname=cls_id,
#             )
    
#     _aps[thresh].append(_ap * 100)
#     _all_precs[thresh].append(_prec)
#     _all_recs[thresh].append(_rec)
#     _tp_plus_fp_cs[thresh].append(_tp_plus_fp_closed_set)
#     _fp_os[thresh].append(_fp_open_set)
#     #logger.info(f"Class: {cls_name}, AP: {_ap * 100:.2f}, Recall: {_rec * 100:.2f}, Precision: {_prec * 100:.2f}")
# print(np.mean(_aps[50][:-1]))
# _tp_plus_fp_cs[50] = _tp_plus_fp_cs[50][:-1]
# _fp_os[50] = _fp_os[50][:-1]
# _all_recs[50] = _all_recs[50][:-1]
# wi_unksniffer = compute_WI_at_many_recall_level(_all_recs, _tp_plus_fp_cs, _fp_os, known_classes=known_classes)