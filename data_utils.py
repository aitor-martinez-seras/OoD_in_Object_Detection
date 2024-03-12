from pathlib import Path
import json
import pickle
from typing import Tuple

import torch
import numpy as np
from PIL import Image
import torchvision.ops as t_ops
from torch.utils.data import DataLoader

from ultralytics.yolo.data.utils import check_det_dataset
from ultralytics.yolo.data.build import build_yolo_dataset, build_dataloader, build_tao_dataset, build_filtered_yolo_dataset
from ultralytics.yolo.utils import DEFAULT_CFG
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data import YOLODataset, BaseDataset
from ultralytics.yolo.data.build import InfiniteDataLoader

def segmentation_to_bbox(segmentation_img: Image, seg_value: int):
    
    segmentation = np.where(np.array(segmentation_img) == seg_value)

    # Bounding Box
    bbox = 0, 0, 0, 0
    if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
        x_min = int(np.min(segmentation[1]))
        x_max = int(np.max(segmentation[1]))
        y_min = int(np.min(segmentation[0]))
        y_max = int(np.max(segmentation[0]))
        bbox = (x_min, y_min, x_max, y_max)
    return bbox


def write_json(an_object, path_to_file: Path):
    print(f"Started writing object {type(an_object)} data into a json file")
    with open(path_to_file, "w") as fp:
        json.dump(an_object, fp)
        print(f"Done writing JSON data into {path_to_file} file")


def read_json(path_to_file: Path):
    # for reading also binary mode is important
    with open(path_to_file, 'rb') as fp:
        an_object = json.load(fp)
        return an_object


def write_pickle(an_object, path_to_file: Path):
    print(f"Started writing object {type(an_object)} data into a .pkl file")
    # store list in binary file so 'wb' mode
    with open(path_to_file, 'wb') as fp:
        pickle.dump(an_object, fp)
        print('Done writing list into a binary file')


def read_pickle(path_to_file: Path):
    # for reading also binary mode is important
    with open(path_to_file, 'rb') as fp:
        an_object = pickle.load(fp)
        return an_object


def create_dataloader(dataset, args):
    """
    Create dataloader with shuffle=False and drop_last=False by default.
    """

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False
    )

    return dataloader


def load_dataset_and_dataloader(dataset_name, data_split: str, batch_size: int,
                                workers: int, owod_task='') -> Tuple[BaseDataset, InfiniteDataLoader]:

    # Get dataset imgs path
    data_dict = check_det_dataset(dataset_name)
    imgs_path = data_dict[data_split]
    type_of_YOLO_dataset = data_dict.get('dataset_class', 'YOLODataset')

    # TODO: Split train dataset into two subsets, one for modeling the in-distribution 
    #   and the other for defining the thresholds using 
    #   https://github.com/ultralytics/ultralytics/blob/437b4306d207f787503fa1a962d154700e870c64/ultralytics/data/utils.py#L586
    
    # Load config
    overrides = {}
    if owod_task:
        overrides["owod_task"] = owod_task
    cfg = get_cfg(DEFAULT_CFG, overrides=overrides)

    # Select dataset
    if 'tao' in dataset_name:
        dataset = build_tao_dataset(
            dataset = build_tao_dataset(
            cfg=cfg,
            img_path=imgs_path,  # Path to the folder containing the images
            batch=batch_size,
            data=data_dict,  # El data dictionary que se puede sacar de data = check_det_dataset(self.args.data)
            mode='test',  # This is for disabling data augmentation
    )
        )
    elif 'coco' in dataset_name:
        if type_of_YOLO_dataset == 'FilteredYOLODataset':
            dataset = build_filtered_yolo_dataset(
                cfg=cfg,
                img_path=imgs_path,  # Path to the folder containing the images
                batch=batch_size,
                data=data_dict,  # El data dictionary que se puede sacar de data = check_det_dataset(self.args.data)
                mode='test',  # This is for disabling data augmentation
            )
        elif type_of_YOLO_dataset == 'YOLODataset':
            dataset = build_yolo_dataset(
                cfg=cfg,
                img_path=imgs_path,  # Path to the folder containing the images
                batch=batch_size,
                data=data_dict,  # El data dictionary que se puede sacar de data = check_det_dataset(self.args.data)
                mode='test',  # This is for disabling data augmentation
            )
        else:
            raise ValueError(f"Wrong dataset class {type_of_YOLO_dataset} in yaml file")
    else:
        raise ValueError(f"Wrong dataset name {dataset_name}")

    dataloader = build_dataloader(
        dataset=dataset,
        batch=batch_size,
        workers=workers,
        shuffle=False,
        rank=-1  # For distributed computing, leave -1 if no distributed computing is done
    )

    return dataset, dataloader


def create_TAO_dataset_and_dataloader(dataset_name, args, data_split: str, batch_size: int, workers: int,
                                       stride: int = 32, fraction: float = 1.0,) -> Tuple[BaseDataset, InfiniteDataLoader]:

    # TODO: En overrides se definirian ciertos parametros que se quieran tocar de la configuracion por defecto,
    # de tal forma que get_cfg() se encarga de coger esa configuracion por defecto y sobreescribirla con 
    # lo que nosotros hayamos definido en overrides. Lo mejor para definir estos overrides es sacarlos tanto de
    # otro archivo de configuracion que nosotros cargemos como queramos (desde un YAML o de un script de python)
    # como sacarlo del argparser
    overrides = {}
    cfg = get_cfg(DEFAULT_CFG, overrides=overrides)

    data_dict = check_det_dataset(dataset_name)
    imgs_path = data_dict[data_split]

    # TODO: Split train dataset into two subsets, one for modeling the in-distribution 
    #   and the other for defining the thresholds using 
    #   https://github.com/ultralytics/ultralytics/blob/437b4306d207f787503fa1a962d154700e870c64/ultralytics/data/utils.py#L586

    dataset = build_tao_dataset(
        cfg=cfg,
        img_path=imgs_path,  # Path to the folder containing the images
        batch=batch_size,
        data=data_dict,  # El data dictionary que se puede sacar de data = check_det_dataset(self.args.data)
        mode='test',  # This is for disabling data augmentation
        rect=False,
        stride=32,
    )

    dataloader = build_dataloader(
        dataset=dataset,
        batch=args.batch_size,
        workers=args.workers,
        shuffle=False,
        rank=-1  # For distributed computing, leave -1 if no distributed computing is done
    )

    return dataset, dataloader


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
