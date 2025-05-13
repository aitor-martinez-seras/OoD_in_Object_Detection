# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops

from typing import List, Tuple

import torch
from torch import Tensor
from torchvision.ops import roi_align

def extract_roi_aligned_features_from_correct_stride(
    ftmaps: List[Tensor],
    boxes: List[Tensor],
    strides: List[Tensor],
    img_shape: List[int],
    device: torch.device,
    extract_all_strides: bool = False
    ) -> List[List[List[Tensor]]]:
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
    :return: List of lists of lists of tensors. The first dimension is the batch size, the second is the stride, 
        and the third has two positions, the bbox index and the feature maps. The tensor of the bbox index has shape [M] and
        the tensor of feature maps has shape [M, C, H, W] where M is the number of bboxes in the stride and C is the number of channels.
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
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    This predictor specializes in object detection tasks, processing model outputs into meaningful detection results
    with bounding boxes and class predictions.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (nn.Module): The detection model used for inference.
        batch (list): Batch of images and metadata for processing.

    Methods:
        postprocess: Process raw model predictions into detection results.
        construct_results: Build Results objects from processed predictions.
        construct_result: Create a single Result object from a prediction.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.detect import DetectionPredictor
        >>> args = dict(model="yolo11n.pt", source=ASSETS)
        >>> predictor = DetectionPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """
        Post-process predictions and return a list of Results objects.

        This method applies non-maximum suppression to raw model predictions and prepares them for visualization and
        further analysis.

        Args:
            preds (torch.Tensor): Raw predictions from the model.
            img (torch.Tensor): Processed input image tensor in model input format.
            orig_imgs (torch.Tensor | list): Original input images before preprocessing.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            (list): List of Results objects containing the post-processed predictions.

        Examples:
            >>> predictor = DetectionPredictor(overrides=dict(model="yolo11n.pt"))
            >>> results = predictor.predict("path/to/image.jpg")
            >>> processed_results = predictor.postprocess(preds, img, orig_imgs)
        """
        save_feats = getattr(self, "save_feats", False)

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

                preds, strides = ops.non_max_suppression_old(
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
                preds = ops.non_max_suppression_old(preds,
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

                preds = ops.non_max_suppression_old(preds,
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
                preds, strides = ops.non_max_suppression_old(
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

            elif self.model.model.extraction_mode == 'ftmaps_and_strides_exact_pos':
                
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
                range8 = torch.arange(s8*s8, device=device)
                range16 = torch.arange(s16*s16, device=device) + s8*s8
                range32 = torch.arange(s32*s32, device=device) + s8*s8 + s16*s16
                strides = torch.cat(
                    (range8,  # 0
                     range16,  # 1
                     range32)  # 2
                )

                # Perform NMS and extract the strides to which the final predicted bboxes belong
                preds, strides = ops.non_max_suppression_old(
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

        else:
        
            preds = ops.non_max_suppression(
                preds,
                self.args.conf,
                self.args.iou,
                self.args.classes,
                self.args.agnostic_nms,
                max_det=self.args.max_det,
                nc=0 if self.args.task == "detect" else len(self.model.names),
                end2end=getattr(self.model, "end2end", False),
                rotated=self.args.task == "obb",
                return_idxs=save_feats,
            )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]

        #results = self.construct_results(preds, img, orig_imgs, **kwargs)
        
        # Codigo de resultados antiguo
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



        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f  # add object features to results

        return results

    def get_obj_feats(self, feat_maps, idxs):
        """Extract object features from the feature maps."""
        from math import gcd

        import torch

        s = gcd(*[x.shape[1] for x in feat_maps])  # find smallest vector length
        obj_feats = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, s, x.shape[1] // s).mean(dim=-1) for x in feat_maps], dim=1
        )  # mean reduce all vectors to same length
        return [feats[idx] if len(idx) else [] for feats, idx in zip(obj_feats, idxs)]  # for each img in batch

    def construct_results(self, preds, img, orig_imgs):
        """
        Construct a list of Results objects from model predictions.

        Args:
            preds (List[torch.Tensor]): List of predicted bounding boxes and scores for each image.
            img (torch.Tensor): Batch of preprocessed images used for inference.
            orig_imgs (List[np.ndarray]): List of original images before preprocessing.

        Returns:
            (List[Results]): List of Results objects containing detection information for each image.
        """
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        """
        Construct a single Results object from one image prediction.

        Args:
            pred (torch.Tensor): Predicted boxes and scores with shape (N, 6) where N is the number of detections.
            img (torch.Tensor): Preprocessed image tensor used for inference.
            orig_img (np.ndarray): Original image before preprocessing.
            img_path (str): Path to the original image file.

        Returns:
            (Results): Results object containing the original image, image path, class names, and scaled bounding boxes.
        """
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])
