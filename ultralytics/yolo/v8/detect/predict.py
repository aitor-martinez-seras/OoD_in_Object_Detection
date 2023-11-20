# Ultralytics YOLO 🚀, AGPL-3.0 license
from typing import List, Tuple

import torch
from torch import Tensor
from torchvision.ops import roi_align

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops


def extract_roi_aligned_features_from_correct_stride(ftmaps: List[Tensor], boxes: List[Tensor], strides: List[Tensor],
                                                     img_shape: List[int], device) -> List[List[Tensor]]:
    """
    Extracts the features of each bounding box from the correct feature map, using RoIAlign. This is needed because
    the bounding boxes are predicted in the original image, but the features are extracted from the feature maps,
    which have a different size. This function extracts the features from the correct feature map for each bounding box.
    :param ftmaps: List of feature maps, each position being ftmaps of one stride. Each element of the list is
        a tensor of shape [N, C, H, W] where N is the batch size, C is the number of channels, and H and W are
        the height and width of the feature map.
    :param boxes: List of tensors, each tensor is a list of bounding boxes for each image in the batch.
        Each tensor has shape [N, 4] where N is the number of bounding boxes, and the 4 elements are the
        coordinates of the bounding box in the original image.
    :param strides: List of tensors, each tensor is a list of the stride of each bounding box in the batch, with the shape [N].
    :param img_shape: Shape of the original image, [H, W].
    :param device: Device to use for the tensors.
    :return: List of lists of tensors, each tensor is the feature map of the corresponding bounding box.
        The first list has as many elements as images in the batch (N), and each element is a list of tensors,
        each position of the second list being the stride (3). Each tensor has shape [M, C, H, W] where M is the
        number of bounding boxes in the image which have been prediced with the corresponding stride.
    """
    ## OPCION 1: Hacer un loop para cada stride, y dentro de ese loop hacer otro loop para cada imagen del batch
    #   Se hace en un solo loop, loopeando los strides y haciendo el roi_align con el ftmap
    #   correspondiente para todo el batch de imagenes a la vez, lo que es más eficiente.
    #   Requiere usar la propiedad de que las cajas las puedes pasar como un tensor que tiene la forma
    #   [K, 5] donde K es el numero de cajas totales entre todas las imagenes del batch y la primera columna
    #   es el indice del batch de la imagen a la que pertenece la caja. De esta forma, si tienes 3 imagenes en el batch
    #   y 10 cajas en total, el tensor de cajas tendria la forma [10, 5] donde la primera columna sería 0, 1 o 2 en funcion
    #   de a que imagen del batch pertenezca la caja.    
    
    # Desenrollo los strides en un tensor
    strides_cat = torch.cat(strides)

    # Desenrollo los indices de a que imagen pertenece cada bbox y los indices de cada bbox dentro de su imagen
    img_idx_for_every_bbox = torch.cat([torch.ones((len(x), 1), device=device) * idx for idx, x in enumerate(strides)])
    box_idx_inside_an_img = torch.cat([torch.arange(len(b), device=device, dtype=torch.int16) for b in boxes])

    # Desenrollo las cajas en un tensor y les añado la info de a que imagen pertenecen en la primera columna
    boxes_cat = torch.cat(boxes)
    boxes_with_img_idx = torch.cat((img_idx_for_every_bbox, boxes_cat), dim=1)  # Add the index of the img (or batch) to the boxes for indexing by image
    batch_size = len(boxes)

    # 

    # Lista donde van las features de cada caja con su stride correcto, para cada imagen del batch
    # Es decir, una lista con tantos elementos como imagenes en el batch, 
    # y en cada en elemento hay otras 3 listas, una por cada stride
    roi_aligned_ftmaps_per_image = [[] for _ in range(batch_size)]
    
    for idx_stride in range(len(ftmaps)):
        # Hay que sacar el indice de cuales cajas que corresponden a este stride
        which_idx_for_current_stride = torch.where(strides_cat == idx_stride)[0]

        # Nos quedamos con los indices de las cajas dentro de su imagen que corresponden a este stride
        idx_of_bbox_inside_img_for_current_stride = box_idx_inside_an_img[which_idx_for_current_stride]
        idx_of_img_for_current_stride = img_idx_for_every_bbox[which_idx_for_current_stride].squeeze()
                                        
        # Explicacion elemento a elemento del RoIAlign:
        # input -> a traves de la list comprenhension recorremos los 3 feature maps de la imagen
        #       en la variable o_ext, la cual indexamos con el indice de la imagen del batch (idx_img)
        #       y le hacemos un unsqueeze para que tenga la forma [1, C, H, W]
        # boxes -> En este caso le pasamos un tensor de la forma [K, 5], ya que el RoIAlign tiene la propiedad 
        #       de que las cajas las puedes pasar como un tensor que tiene la forma [K, 5] donde K es el numero de cajas totales
        #       entre todas las imagenes del batch y la primera columna es el indice del batch de la imagen a la que pertenece la caja.
        # output_size -> define el tamaño de salida de la ROI. Tiene que ser menor que el tamaño de feature maps, y se 
        #       trata de un hyperparametro que hay que ajustar. En este caso lo ponemos a 10x10 por probar.
        # spatial_scale -> es el factor de escala entre las cajas y el feature map. En este caso lo calculamos como el
        #       ancho del feature map entre el ancho de la imagen original ( H/640 )
        roi_aligned_ftmaps_one_stride = roi_align(
            input=ftmaps[idx_stride],
            boxes=boxes_with_img_idx[which_idx_for_current_stride],
            output_size= (10, 10),
            spatial_scale=ftmaps[idx_stride].shape[2]/img_shape[2]
        )
 
        # Ahora hay que volver a asignar cada ftmap de cada caja a su prediccion y stride correspondiente
        for idx_img in range(batch_size):
            idx_of_roi_aligned_ftmaps_to_select_for_current_stride_and_image = torch.where(idx_of_img_for_current_stride == idx_img)[0]
            if idx_of_roi_aligned_ftmaps_to_select_for_current_stride_and_image.shape[0] > 0:
                # First element is the index of the bbox inside the image, second element is the ftmaps corresponding to that img and that stride
                roi_aligned_ftmaps_per_image[idx_img].append(
                    [
                        idx_of_bbox_inside_img_for_current_stride[idx_of_roi_aligned_ftmaps_to_select_for_current_stride_and_image],
                        roi_aligned_ftmaps_one_stride[idx_of_roi_aligned_ftmaps_to_select_for_current_stride_and_image]
                    ]
                )
            else:
                # Empty tensor
                roi_aligned_ftmaps_per_image[idx_img].append([torch.empty(0, device=device), torch.empty(0, device=device)])

    # OPCION 2: Hacer un loop para cada imagen del batch, y dentro de ese loop hacer otro loop para cada stride
    # Con este loop recorremos la dimension de batch
    # for idx_img, one_img_bboxes in enumerate(only_bboxes):

    #     strides_of_each_bbox = [torch.where(strides[idx_img] == idx) for idx in range(number_of_strides)]
    #     aux_list = list()
    #     idx_of_each_bbox_in_batch = [list(range(len(st))) for st in strides_of_each_bbox]
    #     which_stride = strides[idx_img]
    #     idx_of_stride = list(range(len(strides[idx_img])))
    #     idx_of_strides = [list(range(len(st))) for st in strides[idx_img]]

    #     s8_preds = roi_align(
    #         input=torch.unsqueeze(output_extra[0][idx_img], dim=0),
    #         boxes=[one_img_bboxes],
    #         output_size= (10, 10),
    #         spatial_scale=output_extra[0].shape[2]/img.shape[2]
    #     )

    return roi_aligned_ftmaps_per_image  # Sobreescibimos el output_extra con el resultado del roi_align

class DetectionPredictor(BasePredictor):

    def postprocess(self, preds, img, orig_imgs):
        """Postprocesses predictions and returns a list of Results objects."""

        ## preds es tupla de 2 elementos:
        ## 1ero, tensor de (1,84,5040).
        ##  1 es el batch_size (puede ser mas), 84 son el box (4) + las clases (80). 5040 son las boxes (depende del tamaño de la imagen).
        ## 2ndo, lista de 3 elementos. En cada elemento tenemos un tensor. Son los feature map segun Aitor
        # output_extra = preds[1]
        if hasattr(self, 'modo'):
            if self.model.model.modo == 'conv':
                # Con CKA vamos a manejar los feature maps
                # Para la prediccion se usan las 3 escalas de feature maps (8,16,32)
                # y por tanto me he traido las 3 escalas, que vienen en la forma
                # List[[N, CH, H, W]] donde cada posicion de la lista es una de las escalas
                # N es el batch_size, CH es el numero de canales y H y W son el alto y ancho del feature map.
                output_extra = preds[1]
            else:
                output_extra = preds[1][0]
            
            ## Cojo el [0] porque ahora mismo preds[1] es una lista de 2 elementos donde:
            ## el primer elemento es la salida de la red neuronal
            ## el segundo elemento hay 3 items, con lo que creemos que son los feature map.
            preds = preds[0]

            # Strides. We use this variable to mantain the info of which is the scale of each bounding box
            #   in the final predictions. This is needed for using RoIAlign with the correct feature map
            input_size = img.shape[2]
            s8 = input_size // 8
            s16 = input_size // 16
            s32 = input_size // 32
            device = self.device
            strides = torch.cat(
                (torch.zeros((s8*s8), device=device),  # 0
                torch.ones((s16*s16), device=device),  # 1
                torch.ones((s32*s32), device=device) * 2)  # 2
            )

            if self.model.model.modo == 'conv':

                preds, strides = ops.non_max_suppression(
                    preds,
                    self.args.conf,
                    self.args.iou,
                    agnostic=self.args.agnostic_nms,
                    max_det=self.args.max_det,
                    classes=self.args.classes,  # Usually None
                    extra_item=None,
                    strides=strides
                )
                
                roi_aligned_ftmaps_per_image_and_stride = extract_roi_aligned_features_from_correct_stride(
                    ftmaps=output_extra,
                    boxes=[x[:, :4] for x in preds],  # Extracting only the boxes from the predictions
                    strides=strides,
                    img_shape=img.shape,
                    device=device
                )

                # OPCION 2: Hacer un loop para cada imagen del batch, y dentro de ese loop hacer otro loop para cada stride
                # Con este loop recorremos la dimension de batch
                # for idx_img, one_img_bboxes in enumerate(only_bboxes):

                #     strides_of_each_bbox = [torch.where(strides[idx_img] == idx) for idx in range(number_of_strides)]
                #     aux_list = list()
                #     idx_of_each_bbox_in_batch = [list(range(len(st))) for st in strides_of_each_bbox]
                #     which_stride = strides[idx_img]
                #     idx_of_stride = list(range(len(strides[idx_img])))
                #     idx_of_strides = [list(range(len(st))) for st in strides[idx_img]]

                #     s8_preds = roi_align(
                #         input=torch.unsqueeze(output_extra[0][idx_img], dim=0),
                #         boxes=[one_img_bboxes],
                #         output_size= (10, 10),
                #         spatial_scale=output_extra[0].shape[2]/img.shape[2]
                #     )

                output_extra = roi_aligned_ftmaps_per_image_and_stride  # Sobreescibimos el output_extra con el resultado del roi_align

            elif self.model.model.modo == 'logits':
                preds = ops.non_max_suppression(preds,
                                                self.args.conf,
                                                self.args.iou,
                                                agnostic=self.args.agnostic_nms,
                                                max_det=self.args.max_det,
                                                classes=self.args.classes,
                                                extra_item=output_extra)
                output_extra = preds[1]
                # print('++++++++++++ POSTPROCESS ++++++++++++++++')
                # # print(output_extra)
                # print(len(output_extra))
                # for idx, o in enumerate(output_extra):
                #     print(f'Extra item shape: {o.shape}')
                #     print(f'Preds shape: {preds[0][idx].shape}')
                # print('-----------------------------------------------------------------------')

                preds = preds[0]
        
        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            if self.model.model.modo in ['logits', 'conv']:
                results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred, extra_item=output_extra[i]))
            else:
                results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))

        return results


def predict(cfg=DEFAULT_CFG, use_python=False):
    """Runs YOLO model inference on input image(s)."""
    model = cfg.model or 'yolov8n.pt'
    source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
        else 'https://ultralytics.com/images/bus.jpg'

    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    predict()
