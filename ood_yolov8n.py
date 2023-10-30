import time
import os
import json
from pathlib import Path
import pickle
from datetime import datetime

import numpy as np
import torch
from torch.autograd import Variable
from ultralytics import YOLO
from ultralytics.yolo.data.utils import check_det_dataset
from ultralytics.yolo.data.build import build_yolo_dataset, build_dataloader
from ultralytics.yolo.utils import DEFAULT_CFG
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data import YOLODataset
from ultralytics.yolo.engine.results import Results

import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
import torchvision.ops as t_ops
from scipy.optimize import linear_sum_assignment

from ood_utils import get_measures, arg_parser
import log

NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
STORAGE_PATH = Path('storage')
PRUEBAS_ROOT_PATH = Path('pruebas')

############################################################

# Code copied from https://github.com/KingJamesSong/RankFeat

############################################################

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


# TODO: Hay que hacer que esto carge los dataloaders de el dataset In-Distribution y Out-Distribution
def make_id_ood(args, logger):
    """Returns train and ood datasets."""
    # crop = 480

    # val_tx = tv.transforms.Compose([
    #     tv.transforms.Resize((crop, crop)),
    #     tv.transforms.ToTensor(),
    #     tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])

    # in_set = tv.datasets.ImageFolder(args.in_datadir, val_tx)
    # out_set = tv.datasets.ImageFolder(args.out_datadir, val_tx)

    #in_set = 

    logger.info(f"Using an in-distribution set with {len(in_set)} images.")
    logger.info(f"Using an out-of-distribution set with {len(out_set)} images.")

    in_loader = torch.utils.data.DataLoader(
        in_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    ood_loader = torch.utils.data.DataLoader(
        out_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    return in_set, out_set, in_loader, ood_loader


def create_YOLO_dataset_and_dataloader(dataset_yaml_file_name_or_path, batch_size: int, data_split: str, workers: int,
                                       stride: int = 32, fraction: float = 1.0,):

    # TODO: En overrides se definirian ciertos parametros que se quieran tocar de la configuracion por defecto,
    # de tal forma que get_cfg() se encarga de coger esa configuracion por defecto y sobreescribirla con 
    # lo que nosotros hayamos definido en overrides. Lo mejor para definir estos overrides es sacarlos tanto de
    # otro archivo de configuracion que nosotros cargemos como queramos (desde un YAML o de un script de python)
    # como sacarlo del argparser
    overrides = dict()
    cfg = get_cfg(DEFAULT_CFG, overrides=overrides)

    data_dict = check_det_dataset(dataset_yaml_file_name_or_path)
    imgs_path = data_dict[data_split]

    # TODO: Split train dataset into two subsets, one for modeling the in-distribution 
    #   and the other for defining the thresholds using 
    #   https://github.com/ultralytics/ultralytics/blob/437b4306d207f787503fa1a962d154700e870c64/ultralytics/data/utils.py#L586

    dataset = build_yolo_dataset(
        cfg=cfg,
        img_path=imgs_path,  # Path to the folder containing the images
        batch=batch_size,
        data=data_dict,  # El data dictionary que se puede sacar de data = check_det_dataset(self.args.data)
        mode='test',  # This is for disabling data augmentation
        rect=False,
        stride=32,
    )

    # from ultralytics.yolo.utils import colorstr
    # dataset = YOLODataset(
    #     img_path=imgs_path,
    #     imgsz=cfg.imgsz,
    #     batch_size=batch_size,
    #     augment=False,  # We dont want to augment the data
    #     hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
    #     rect=cfg.rect,  # rectangular batches
    #     cache=cfg.cache or None,
    #     single_cls=cfg.single_cls or False,
    #     stride=int(stride),
    #     #pad=0.0 if mode == 'train' else 0.5,
    #     pad=0.5,
    #     prefix=colorstr(f'test: '),
    #     use_segments=cfg.task == 'segment',
    #     use_keypoints=cfg.task == 'pose',
    #     classes=cfg.classes,
    #     data=data_dict,
    #     fraction=fraction)

    dataloader = build_dataloader(
        dataset=dataset,
        batch=dataset.batch_size,
        workers=cfg.workers,
        shuffle=False,
        rank=-1  # For distributed computing, leave -1 if no distributed computing is done
    )

    return dataset, dataloader


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

    if idx_of_batch == int(number_of_batches * 0.1):
        logger.info(f"10%: Procesing batch {idx_of_batch} of {number_of_batches}")
    elif idx_of_batch == int(number_of_batches * 0.25):
        logger.info(f"25%: Procesing batch {idx_of_batch} of {number_of_batches}")
    elif idx_of_batch == int(number_of_batches * 0.5):
        logger.info(f"50%: Procesing batch {idx_of_batch} of {number_of_batches}")
    elif idx_of_batch == int(number_of_batches * 0.75):
        logger.info(f"75%: Procesing batch {idx_of_batch} of {number_of_batches}")


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


def plot_results(class_names, results, valid_preds_only: bool, origin_of_idx: int, ood_decision=None, ood_method=None):
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
        plt.savefig(prueba_ahora_path / f'{(origin_of_idx + img_idx):03}.png', dpi=300)
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


def generate_predictions_with_ood_labeling(dataloader, model, device, logger, ood_thresholds, ood_method, temper=None):
    
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
            # Iteramos cada bbox
            for idx_bbox in range(len(res.boxes.cls)):

                # Extraemos la clase predicha y los logits
                cls_idx = int(res.boxes.cls[idx_bbox].cpu())
                logits = res.extra_item[idx_bbox][4:].cpu()

                if ood_method == 'msp':
                    # Coger el cls_idx es como hacer el .max()
                    if logits[cls_idx] < ood_thresholds[cls_idx]:
                        ood_decision[idx_img].append(0)
                    else:
                        ood_decision[idx_img].append(1)

                elif ood_method == 'Energy':
                    energy_score = temper * torch.logsumexp(logits / temper, dim=0).item()
                    if energy_score < ood_thresholds[cls_idx]:
                        ood_decision[idx_img].append(0)
                    else:
                        ood_decision[idx_img].append(1)
        
        plot_results(
            model.names,
            results,
            valid_preds_only=False,
            origin_of_idx=idx_of_batch*dataloader.batch_size,
            ood_decision=ood_decision,
            ood_method=ood_method
        )
        
        if idx_of_batch > 4:
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


# CKA Score
def iterate_data_cka(data_loader, model, device, logger):

    import cka
    
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

        ### Acumular el CKA score de cada caja en la clase correspondiente ###
        # Recorremos cada imagen
        for res in results:
            # Recorremos cada caja predicha valida
            for valid_idx_one_bbox in res.valid_preds:
                # Extraemos el indice de la clase predicha para la caja actual
                cls_idx_one_bbox = int(res.boxes.cls[valid_idx_one_bbox].cpu())

                # Compute CKA score
                # TODO: De momento esta hecho para un solo feature map, pero habria que hacerlo de tal
                #   forma que para cada bbox se elija el feature map con el que ha sido realizada la prediccion.
                #   Esto probablemente se tenga que realizar en la funcion ops.non_max_supression() de
                #   /groups/tri110414/yolo-pruebas/ultralytics/yolo/v8/detect/predict.py
                print()
                import cka_google
                idx_bboxes = [1,3,4,5]
                activations = []
                for idx_box in idx_bboxes:
                    acts = res.extra_item[0][idx_box]
                    # activations.append(np.expand_dims(np.mean(acts.cpu().numpy(), axis=(1,2)), axis=0))
                    activations.append(np.expand_dims(torch.flatten(acts, start_dim=0).cpu().numpy(), axis=0))
                min_len = 1000
                max_len = 1050
                cka_all_res = np.zeros((len(activations), max_len))
                for idx_bbox_i in range(len(activations)):
                    for idx_bbox_j in range(idx_bbox_i, len(activations)):
                        for k in range(min_len, max_len):
                            cka_all_res[idx_bbox_i][k] = cka.linear_CKA(
                                np.repeat(activations[idx_bbox_i], k, axis=0), np.repeat(activations[idx_bbox_j], k, axis=0)
                            )
                            cka_all_res[idx_bbox_i][k] = cka_google.feature_space_linear_cka(
                                np.repeat(activations[idx_bbox_i], k, axis=0), np.repeat(activations[idx_bbox_j], k, axis=0)
                            )
                
                for cka_one_res in cka_all_res:
                    cka_one_res = np.nan_to_num(cka_one_res, copy=True, nan=0)
                    plt.plot(cka_one_res)

                plt.xlim([min_len, max_len])
                plt.ylim([-1,1])
                plt.ylabel('CKA Values (-1 to 1)')
                plt.xlabel('Number of repetitions')
                plt.savefig('prueba_cka_all.pdf')
                plt.close()

                idx_bbox0 = 1
                idx_bbox1 = 2
                acts0 = res.extra_item[0][idx_bbox0]
                acts1 = res.extra_item[0][idx_bbox1]
                avg_acts0 = np.expand_dims(np.mean(acts0.cpu().numpy(), axis=(1,2)), axis=0)
                avg_acts1 = np.expand_dims(np.mean(acts1.cpu().numpy(), axis=(1,2)), axis=0)
                min_len = 1000
                max_len = 1050
                cka_res = np.zeros(max_len)
                for k in range(min_len, max_len):
                    cka_res[k] = cka.linear_CKA(np.repeat(avg_acts0, k, axis=0), np.repeat(avg_acts1, k, axis=0))
                cka_res = np.nan_to_num(cka_res, copy=True, nan=0)
                plt.plot(cka_res)
                plt.xlim([min_len, max_len])
                plt.ylim([-1,1])
                plt.ylabel('CKA Values (-1 to 1)')
                plt.xlabel('Number of repetitions')
                plt.savefig('prueba_cka.pdf')
                plt.close()

                # Acumulamos en la clase correspondiente el Energy score
                # Para ello hay que multiplicar los logits por la temperatura y hacer el logsumexp
                #all_scores[cls_idx_one_bbox].append(cka_score).item())

    return all_scores


def generate_ood_thresholds(in_scores: list, tpr: float, per_class: bool, logger) -> np.array:
    """
    Generate the thresholds for each class using the in-distribution scores.
    If per_class=True, in_scores must be a list of lists,
      where each list is the list of scores for each class.
    tpr must be in the range [0, 1]
    """
    if per_class:
        ood_thresholds = np.zeros(len(in_scores))
        for idx, cl_scores in enumerate(in_scores):
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
                                                            args.ood_method, temper=args.temperature_energy)
        
    elif args.ood_method == 'CKA':

        # In scores
        if args.load_in_scores:
            logger.info("Loading in-distribution scores from disk...")
            in_scores = read_json(Path(STORAGE_PATH / 'CKA_in_scores.json'))
        else:
            logger.info("Processing in-distribution data...")
            in_scores = iterate_data_cka(in_loader, model, device, logger)
            # Guardo los resultados en disco para no repetir el calculo
            in_scores_serializable = [[float(score) for score in cl] for cl in in_scores]
            write_json(in_scores_serializable, STORAGE_PATH / 'CKA_in_scores.json')
            write_pickle(in_scores, STORAGE_PATH / 'CKA_in_scores.pkl')
        
        ood_thresholds = generate_ood_thresholds(in_scores, tpr=0.95, per_class=True, logger=logger)

        # OoD scores
        logger.info("Processing out-of-distribution data...")
        ood_scores = generate_predictions_with_ood_labeling(ood_loader, model, device, logger, ood_thresholds,
                                                            args.ood_method)

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
    gpu_number = str(2)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_number
    logger.warning(f'CUDA_VISIBLE_DEVICES = {gpu_number}')

    # TODO: Unused till we implement GradNorm
    if args.ood_method == 'GradNorm':
        args.batch = 1

    # Load ID data and OOD data
    # ind_dataset, ind_dataloader = create_YOLO_dataset_and_dataloader('coco128_custom.yaml', batch_size=args.batch_size)
    #ind_dataset, ind_dataloader = create_YOLO_dataset_and_dataloader('coco128_custom.yaml', batch_size=5)
    ind_dataset, ind_dataloader = create_YOLO_dataset_and_dataloader(
        'coco.yaml',
        batch_size=args.batch_size,
        data_split='val',
        workers=args.num_workers
    )
    ood_dataset, ood_dataloader = create_YOLO_dataset_and_dataloader(
        'VisDrone.yaml', 
        batch_size=args.batch_size,
        data_split='val',
        workers=args.num_workers
    )
    # in_set, out_set, in_loader, ood_loader = make_id_ood(args, logger)

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
                        choices=['MSP', 'ODIN', 'Energy', 'Mahalanobis', 'GradNorm','RankFeat','React', 'CKA'],
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