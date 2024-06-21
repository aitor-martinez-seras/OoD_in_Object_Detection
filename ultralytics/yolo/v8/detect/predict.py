# Ultralytics YOLO 🚀, AGPL-3.0 license
from typing import List, Tuple

import torch
from torch import Tensor
from torchvision.ops import roi_align

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops


def extract_roi_aligned_features_from_correct_stride(
    ftmaps: List[Tensor],
    boxes: List[Tensor],
    strides: List[Tensor],
    img_shape: List[int],
    device: torch.device,
    extract_all_strides: bool = False
    ) -> List[List[Tensor]]:
    """
    Extracts the features of each bounding box from the correct feature map, using RoIAlign. This is needed because
    the bounding boxes are predicted in the original image, but the features are extracted from the feature maps,
    which have a different size. This function extracts the features from the correct feature map for each bounding box.
    :param ftmaps: List of feature maps, each position being all ftmaps of one stride, i.e., the list contains the dimension S (stride).
        Each element of the list is a tensor of shape [N, C, H, W] where N is the batch size, C is the number of channels, 
        and H and W are the height and width of the feature map.
    :param boxes: List of tensors, each tensor is composed of bounding boxes for each image in the batch.
        The list has length N (the batch lenght) and each tensor has shape [M, 4] where M is the number of bounding boxes, 
        and the 4 elements are the coordinates of the bounding box in the original image in xyxy format (x1, y1, x2, y2).
    :param strides: List of tensors, each tensor is a list of the stride of each bounding box in the batch, with the shape [N].
    :param img_shape: Shape of the original image, [H, W].
    :param device: Device to use for the tensors.
    :return: List of lists of tensors, each tensor is the feature map of the corresponding bounding box.
        The first list has as many elements as images in the batch (N), and each element is a list of tensors,
        each position of the second list being the stride (3). Each tensor has shape [M, C, H, W] where M is the
        number of bounding boxes in the image which have been prediced with the corresponding stride.
    """
    ### OPTIMIZED AND IMPROVED VERSION ###

    batch_size = len(boxes)
    strides_cat = torch.cat(strides)
    img_indices = torch.cat([torch.full((len(b),), i, dtype=torch.long, device=device) for i, b in enumerate(boxes)])
    boxes_cat = torch.cat(boxes)
    boxes_with_img_indices = torch.cat([img_indices.unsqueeze(1), boxes_cat], dim=1)

    #roi_aligned_features = [[[] for _ in range(len(ftmaps))] for _ in range(batch_size)]
    roi_aligned_features_per_image_per_stride = [[[[] for _ in range(2)] for _ in range(len(ftmaps))] for _ in range(batch_size)]

    for stride_idx, ftmap in enumerate(ftmaps):
        
        # Extract the boxes relevant for this stride
        if extract_all_strides:
            # All boxes are relevant
            relevant_boxes = boxes_with_img_indices
            stride_mask = torch.full((relevant_boxes.shape[0],), True, dtype=torch.bool, device=device)
        else:
            stride_mask = strides_cat == stride_idx
            relevant_boxes = boxes_with_img_indices[stride_mask]

        if relevant_boxes.shape[0] == 0:
            aligned_features = torch.empty(0, device=device)
        else:
            # Extract features using RoIAlign for all relevant boxes at once
            aligned_features = roi_align(
                input=ftmap,
                boxes=relevant_boxes,
                output_size=(1, 1),  # Example fixed output size
                spatial_scale=ftmap.shape[-1] / img_shape[1],  # Width ratio
                aligned=False
            )

        # Distribute features back to corresponding images and positions
        # for idx_img in range(batch_size):
        #     img_mask = relevant_boxes[:, 0] == idx_img
        #     if img_mask.any():
        #         roi_aligned_features[idx_img][stride_idx].append(aligned_features[img_mask].unbind(0))

        for idx_img in range(batch_size):
            img_mask = boxes_with_img_indices[:, 0] == idx_img
            # Extract the indices of the boxes for this image and stride
            img_mask_for_current_stride = img_mask[stride_mask]
            stride_mask_for_current_img = stride_mask[img_mask]
            #if img_mask_for_current_stride.any():
            idx_in_img = torch.arange(sum(img_mask), device=device, dtype=torch.int16)  # Get the original indices of the boxes in this image
            if not extract_all_strides:
                idx_in_img = idx_in_img[stride_mask_for_current_img]
            roi_aligned_features_per_image_per_stride[idx_img][stride_idx][0] = idx_in_img  # Indices of boxes for this image and stride
            roi_aligned_features_per_image_per_stride[idx_img][stride_idx][1] = aligned_features[img_mask_for_current_stride, ...]  # Feature maps for these indices

    return roi_aligned_features_per_image_per_stride


class DetectionPredictor(BasePredictor):

    def postprocess(self, preds, img, orig_imgs):
        """Postprocesses predictions and returns a list of Results objects."""

        # Preds:
        #   - Si viene sin output_extra: 
        #       - preds[0] es un tensor de (N, n_cls + 4, 8000) -> Las predicciones
        #       - preds[1] es una lista con 3 tensores de (N, 144, H, W)  -> No se que son...
        #   - Si viene con output_extra:
        #       - preds[0] es el tensor sin output_extra (es lo mostrado arriba, con las preds en la posicion 0)
        #       - preds[1] es el output_extra:
        #           - Si es 'roi_aligned_ftmaps' O 'all_ftmaps': lista de 3 tensores de (N, C, H, W) -> Los feature maps
        #           - Si es 'logits': tensor de  -> Los logits
        #           - Si es 'ftmaps_and_strides': (feature_maps, strides) -> Los feature maps y la asignacion de las predicciones al stride

        ### Execution for internal information extraction ###
        ood_info_retrieval_mode = hasattr(self.model.model, 'extraction_mode')
        if ood_info_retrieval_mode:  #in ['roi_aligned_ftmaps', 'logits', 'all_ftmaps', 'ftmaps_and_strides']:
            
            if self.model.model.model[-1].output_values_before_sigmoid:
                output_extra = preds[0][1]  # Los valores antes de la sigmoide
            else:
                output_extra = preds[1]  # Los feature maps o logits
            preds = preds[0][0]  # Las predicciones
            
            if self.model.model.extraction_mode == 'roi_aligned_ftmaps':
                # Para la prediccion se usan las 3 escalas de feature maps (8,16,32)
                # y por tanto me he traido las 3 escalas, que vienen en la forma
                # List[[N, CH, H, W]] donde cada posicion de la lista es una de las escalas [8,16,32]
                # N es el batch_size, CH es el numero de canales y H y W son el alto y ancho del feature map.

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
                    img_shape=img.shape[2:],
                    device=device
                )

                output_extra = roi_aligned_ftmaps_per_image_and_stride  # Sobreescibimos el output_extra con el resultado del roi_align

            elif self.model.model.extraction_mode == 'logits':
                if not self.model.model.model[-1].output_values_before_sigmoid:
                    output_extra = output_extra[0]  # Los logits
                preds = ops.non_max_suppression(preds,
                                                self.args.conf,
                                                self.args.iou,
                                                agnostic=self.args.agnostic_nms,
                                                max_det=self.args.max_det,
                                                classes=self.args.classes,
                                                extra_item=output_extra)
                output_extra = preds[1]
                preds = preds[0]

            elif self.model.model.extraction_mode == 'all_ftmaps':
                # Tenemos que hacer que para cada prediccion (dimension N) haya una lista de 3 tensores, uno por cada escala
                output_extra = [list(batch) for batch in zip(*output_extra)]
                # # El codigo equivalente al one-liner de arriba es:
                # new_output_extra = [[[],[],[]] for _ in range(len(preds))]  # shape (N, 3) pero con listas
                # for idx_batch in range(len(preds)):
                #     for idx_lvl in range(len(output_extra)):
                #         new_output_extra[idx_batch][idx_lvl] = output_extra[idx_lvl][idx_batch]
                # output_extra = new_output_extra

                preds = ops.non_max_suppression(preds,
                                                self.args.conf,
                                                self.args.iou,
                                                agnostic=self.args.agnostic_nms,
                                                max_det=self.args.max_det,
                                                classes=self.args.classes)

            elif self.model.model.extraction_mode == 'ftmaps_and_strides':
                
                # Create a list of lists, where first dimension is N and second is S, the stride
                # Each element of the second list is a tensor of shape [C, H, W] where C is the number of channels
                output_extra = [list(batch) for batch in zip(*output_extra)]

                # Strides. We use this variable to mantain the info of which is the scale of each predicted bounding box
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

                # Perform NMS and extract the strides to which the final predicted bboxes belong
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

                # Merge the feature maps and the strides into a list of lists
                # where the first list has as many elements as images in the batch (N),
                # and each element is list of two elements, the feature maps and the strides
                output_extra = list(zip(output_extra, strides))

            
        ### Standard execution ###
        else:
            preds = ops.non_max_suppression(preds[0],
                                            self.args.conf,
                                            self.args.iou,
                                            agnostic=self.args.agnostic_nms,
                                            max_det=self.args.max_det,
                                            classes=self.args.classes)

        
        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            if ood_info_retrieval_mode:  # True if we are retrieving logits or feature maps from the model
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
