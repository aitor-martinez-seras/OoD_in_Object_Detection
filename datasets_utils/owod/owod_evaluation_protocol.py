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

import matplotlib.pyplot as plt
import torch
from torch.distributions.weibull import Weibull
from torch.distributions.transforms import AffineTransform
from torch.distributions.transformed_distribution import TransformedDistribution
# from fvcore.common.file_io import PathManager

# from detectron2.data import MetadataCatalog
# from detectron2.utils import comm

# from detectron2.evaluation.evaluator import DatasetEvaluator
import json
np.set_printoptions(threshold=sys.maxsize)


# f = open('./datasets/t1/annotations/test.json', 'r')
# ground_truth = json.load(f)
# f.close()

# mapping ={}
# for each in ground_truth['images']:
#     mapping[each['id']]=each['file_name'].split('.')[0]

OWOD_FOLDER_PATH = Path(__file__).parent
UNKNOWN_CLASS_INDEX = 80

# class PascalVOCDetectionEvaluator(DatasetEvaluator):
#     """
#     Evaluate Pascal VOC style AP for Pascal VOC dataset.
#     It contains a synchronization, therefore has to be called from all ranks.

#     Note that the concept of AP can be implemented in different ways and may not
#     produce identical results. This class mimics the implementation of the official
#     Pascal VOC Matlab API, and should produce similar but not identical results to the
#     official API.
#     """

#     def __init__(self, dataset_name, cfg=None):
#         """
#         Args:
#             dataset_name (str): name of the dataset, e.g., "voc_2007_test"
#         """
#         self._dataset_name = dataset_name
#         meta = MetadataCatalog.get(dataset_name)
#         self._anno_file_template = os.path.join('./datasets', "Annotations", "{}.xml")
#         self._image_set_path = os.path.join('./split', "all_task_test.txt")
#         self._class_names = meta.thing_classes
#         self._is_2007 = False
#         # self._is_2007 = meta.year == 2007
#         self._cpu_device = torch.device("cpu")
#         logger = logging.getLogger(__name__)
#         if cfg is not None:
#             self.prev_intro_cls = cfg.TEST.PREV_INTRODUCED_CLS
#             self.curr_intro_cls = cfg.TEST.CUR_INTRODUCED_CLS
#             self.total_num_class = cfg.MODEL.RandBox.NUM_CLASSES
#             self.unknown_class_index = self.total_num_class - 1
#             self.num_seen_classes = self.prev_intro_cls + self.curr_intro_cls
#             self.known_classes = self._class_names[:self.num_seen_classes]

#     def reset(self):
#         self._predictions = defaultdict(list)  # class name -> list of prediction strings



#     def process(self, inputs, outputs):
#         for input, output in zip(inputs, outputs):
#             image_id = input["image_id"]
#             instances = output["instances"].to(self._cpu_device)
#             boxes = instances.pred_boxes.tensor.numpy()
#             scores = instances.scores.tolist()
#             classes = instances.pred_classes.tolist()
#             threshold = 0.15
#             for box, score, cls in zip(boxes, scores, classes):
#                 if score < threshold:
#                     continue
#                 if cls == -100:
#                     continue
#                 xmin, ymin, xmax, ymax = box
#                 # The inverse of data loading logic in `datasets/pascal_voc.py`
#                 xmin += 1
#                 ymin += 1
#                 self._predictions[cls].append(
#                     f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
#                 )

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
            if cls_id == UNKNOWN_CLASS_INDEX and len(rec)>0:
                p = precisions[iou][cls_id][min(range(len(rec)), key=lambda i: abs(rec[i] - recall_level))]
                prec.append(p)
        if len(prec) > 0:
            precs[iou] = np.mean(prec)
        else:
            precs[iou] = 0
    return precs

def compute_WI_at_many_recall_level(recalls, tp_plus_fp_cs, fp_os, known_classes):
    wi_at_recall = {}
    for r in range(1, 10):
        r = r/10
        wi = compute_WI_at_a_recall_level(recalls, tp_plus_fp_cs, fp_os, recall_level=r, known_classes=known_classes)
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

    # WI is calculated for all_test?? Or only for WI split?
    wi = compute_WI_at_many_recall_level(all_recs, tp_plus_fp_cs, fp_os, known_classes=known_classes)
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
    if False:
        # TODO: Voy a tener que llevar un trackeo de las classes introducidas previamente y las actuales
        if self.prev_intro_cls > 0:
            # logger.info("\nPrev class AP__: " + str(np.mean(avg_precs[:self.prev_intro_cls])))
            logger.info("Prev class AP50: " + str(np.mean(aps[50][:self.prev_intro_cls])))
            logger.info("Prev class Precisions50: " + str(np.mean(precs[50][:self.prev_intro_cls])))
            logger.info("Prev class Recall50: " + str(np.mean(recs[50][:self.prev_intro_cls])))
            print("Prev class AP50: " + str(np.mean(aps[50][:self.prev_intro_cls])))
            print("Prev class Precisions50: " + str(np.mean(precs[50][:self.prev_intro_cls])))
            print("Prev class Recall50: " + str(np.mean(recs[50][:self.prev_intro_cls])))

            # logger.info("Prev class AP75: " + str(np.mean(aps[75][:self.prev_intro_cls])))

        # logger.info("\nCurrent class AP__: " + str(np.mean(avg_precs[self.prev_intro_cls:self.curr_intro_cls])))
        logger.info("Current class AP50: " + str(np.mean(aps[50][self.prev_intro_cls:self.prev_intro_cls + self.curr_intro_cls])))
        logger.info("Current class Precisions50: " + str(np.mean(precs[50][self.prev_intro_cls:self.prev_intro_cls + self.curr_intro_cls])))
        logger.info("Current class Recall50: " + str(np.mean(recs[50][self.prev_intro_cls:self.prev_intro_cls + self.curr_intro_cls])))
        print("Current class AP50: " + str(np.mean(aps[50][self.prev_intro_cls:self.prev_intro_cls + self.curr_intro_cls])))
        print("Current class Precisions50: " + str(np.mean(precs[50][self.prev_intro_cls:self.prev_intro_cls + self.curr_intro_cls])))
        print("Current class Recall50: " + str(np.mean(recs[50][self.prev_intro_cls:self.prev_intro_cls + self.curr_intro_cls])))
        # logger.info("Current class AP75: " + str(np.mean(aps[75][self.prev_intro_cls:self.curr_intro_cls])))

    # logger.info("\nKnown AP__: " + str(np.mean(avg_precs[:self.prev_intro_cls + self.curr_intro_cls])))
    logger.info("Known AP50: " + str(np.mean(aps[50][:-1])))
    logger.info("Known Precisions50: " + str(np.mean(precs[50][:-1])))
    logger.info("Known Recall50: " + str(np.mean(recs[50][:-1])))
    # print("Known AP50: " + str(np.mean(aps[50][:-1])))
    # print("Known Precisions50: " + str(np.mean(precs[50][:-1])))
    # print("Known Recall50: " + str(np.mean(recs[50][:-1])))
    # logger.info("Known AP75: " + str(np.mean(aps[75][:self.prev_intro_cls + self.curr_intro_cls])))
    # logger.info("Known AP50: " + str(np.mean(aps[50][:self.prev_intro_cls + self.curr_intro_cls])))
    # logger.info("Known Precisions50: " + str(np.mean(precs[50][:self.prev_intro_cls + self.curr_intro_cls])))
    # logger.info("Known Recall50: " + str(np.mean(recs[50][:self.prev_intro_cls + self.curr_intro_cls])))
    # print("Known AP50: " + str(np.mean(aps[50][:self.prev_intro_cls + self.curr_intro_cls])))
    # print("Known Precisions50: " + str(np.mean(precs[50][:self.prev_intro_cls + self.curr_intro_cls])))
    # print("Known Recall50: " + str(np.mean(recs[50][:self.prev_intro_cls + self.curr_intro_cls])))

    # logger.info("\nUnknown AP__: " + str(avg_precs[-1]))
    logger.info("Unknown AP50: " + str(aps[50][-1]))
    logger.info("Unknown Precisions50: " + str(precs[50][-1]))
    logger.info("Unknown Recall50: " + str(recs[50][-1]))
    # print("Unknown AP50: " + str(aps[50][-1]))
    # print("Unknown Precisions50: " + str(precs[50][-1]))
    # print("Unknown Recall50: " + str(recs[50][-1]))
    # logger.info("Unknown AP75: " + str(aps[75][-1]))

    # logger.info("R__: " + str(['%.1f' % x for x in list(np.mean([x for _, x in recs.items()], axis=0))]))
    # logger.info("R50: " + str(['%.1f' % x for x in recs[50]]))
    # logger.info("R75: " + str(['%.1f' % x for x in recs[75]]))
    #
    # logger.info("P__: " + str(['%.1f' % x for x in list(np.mean([x for _, x in precs.items()], axis=0))]))
    # logger.info("P50: " + str(['%.1f' % x for x in precs[50]]))
    # logger.info("P75: " + str(['%.1f' % x for x in precs[75]]))

    # Info for paper
    known_ap50 = np.mean(aps[50][:-1])
    unknown_ap50 = aps[50][-1]
    unknown_f1 = 2 * (precs[50][-1] * recs[50][-1]) / (precs[50][-1] + recs[50][-1])
    unknown_precision = precs[50][-1]
    unknown_recall = recs[50][-1]
    wilderness_impact_recall_08 = wi[0.8][50]
    total_num_unk_det_as_known = total_num_unk_det_as_known[50]
    results_dict = {
        'mAP': known_ap50,
        'U-AP': unknown_ap50,
        'U-F1': unknown_f1,
        'U-PRE': unknown_precision,
        'U-REC': unknown_recall,
        'A-OSE': total_num_unk_det_as_known,
        'WI-08': wilderness_impact_recall_08,
    }
    logger.info('Summary:')
    logger.info('-----------------')    
    logger.info('Known mAP50 [%]: ' + str(known_ap50))
    logger.info('Unknown AP50 [%]: ' + str(unknown_ap50))
    logger.info('Unknown F1 score [%]: ' + str(unknown_f1))
    logger.info('Unknown Precision [%]: ' + str(unknown_precision))
    logger.info('Unknown Recall [%]: ' + str(unknown_recall))
    logger.info('A-OSE: ' + str(total_num_unk_det_as_known))
    logger.info('Wilderness Impact Recall 0.8 [%]: ' + str(wilderness_impact_recall_08))
    logger.info('-----------------')    
    # logger.info('-----------------')
    # logger.info('Known mAP50 [%]: ' + str(np.mean(aps[50][:-1])))
    # logger.info('Unknown AP50 [%]: ' + str(aps[50][-1]))
    # logger.info('Unknown F1 score [%]: ' + str(2 * (precs[50][-1] * recs[50][-1]) / (precs[50][-1] + recs[50][-1])))
    # logger.info('Unknown Precision [%]: ' + str(precs[50][-1]))
    # logger.info('Unknown Recall [%]: ' + str(recs[50][-1]))
    # logger.info('A-OSE: ' + str(total_num_unk_det_as_known))
    # logger.info('Wilderness Impact Recall 0.8 [%]: ' + str(wi[0.8]))
    # logger.info('-----------------')
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
    class_mapping = {class_name: idx if class_name != 'unknown' else 80 for idx, class_name in enumerate(class_names)}
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
    #     difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
    #     # difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
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
    rec = tp / float(npos)  # recall = TP / (TP + FN)
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
    #     difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
    #     det = [False] * len(R)
    #     n_unk = n_unk + sum(~difficult)
    #     unknown_class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}
    # TODO: Aqui tengo que asegurarme de haber designado como unknown aquellas classes que no son conocidas
    unknown_class_recs = {}
    n_unk = 0
    for target in all_targets:
        #if target['img_name'] in imagenames:
        imagename = target['img_name']
        current_class_gt_mask = target['cls'] == 80  # 80 es el indice de unknown
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

    # print(precision)
    # print(recall)