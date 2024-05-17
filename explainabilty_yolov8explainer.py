from pathlib import Path
from datetime import datetime
from typing import Literal, List
import os

from tap import Tap

import numpy as np
from ultralytics import YOLO
# from object_detection_CAM_xai.xai.common.object_detection_models import YOLOv8CAM
# from object_detection_CAM_xai.xai.common._utils import process_image, draw_detections, save_image, renormalize_cam_in_bounding_boxes
# from pytorch_grad_cam import EigenCAM, AblationCAM, ScoreCAM,GradCAM  # Grad-CAM implementation
# from pytorch_grad_cam.utils.image import show_cam_on_image

from YOLOv8_Explainer import yolov8_heatmap
from YOLOv8_Explainer.utils import save_images

print('---------------------------------')
print('Explainability YOLOv8 Explainer')
print('---------------------------------')

# GPU
gpu_number = str(2)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_number
device = 'cuda:0'

# Set model
expl_method = "ScoreCAM"  # GradCAM , HiResCAM, GradCAMPlusPlus, XGradCAM , LayerCAM, EigenGradCAM and EigenCAM
#targ_layers = [15, 18, 21]
targ_layers = [15]
renormalize = False
show_box = True
show_image = False
if show_box:
    show_image = True
model = yolov8_heatmap(
    weight="runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt",
    method=expl_method,
    layer=targ_layers,
    ratio=0.05,
    renormalize=renormalize,
    show_box=show_box,
)

# Load image
path = "/groups/tri110414/datasets/coco/images/val2017/"
image_name = "000000041888.jpg"
img_path = "/groups/tri110414/datasets/coco/images/val2017/000000041888.jpg"
img_paths = [
"/groups/tri110414/datasets/coco/images/val2017/000000041888.jpg",
"/groups/tri110414/datasets/coco/images/val2017/000000555705.jpg",
"/groups/tri110414/datasets/coco/images/val2017/000000500663.jpg",
"/groups/tri110414/datasets/coco/images/val2017/000000418281.jpg",
"/groups/tri110414/datasets/coco/images/val2017/000000025560.jpg",
"/groups/tri110414/datasets/coco/images/val2017/000000185250.jpg",
"/groups/tri110414/datasets/coco/images/val2017/000000572517.jpg",
"/groups/tri110414/datasets/coco/images/val2017/000000516316.jpg",
"/groups/tri110414/datasets/coco/images/val2017/000000382088.jpg",
"/groups/tri110414/datasets/coco/images/val2017/000000476258.jpg",
"/groups/tri110414/datasets/coco/images/val2017/000000515445.jpg",
"/groups/tri110414/datasets/coco/images/val2017/000000173383.jpg",
"/groups/tri110414/datasets/coco/images/val2017/000000063154.jpg",
"/groups/tri110414/datasets/coco/images/val2017/000000551215.jpg",
"/groups/tri110414/datasets/coco/images/val2017/000000544519.jpg",
"/groups/tri110414/datasets/coco/images/val2017/000000213086.jpg"
]
imagelist = model(images=img_paths, return_type='Image', show_image=show_image)

layers_str = "_".join([str(i) for i in targ_layers])
if renormalize:
    save_images(imagelist, folder='explainability_images', name=f"D_yolov8_explainer_{expl_method}_{layers_str}_renormalized.jpg")
else:
    save_images(imagelist, folder='explainability_images', name=f"D_yolov8_explainer_{expl_method}_{layers_str}.jpg")
print()
# display_images(imagelist)

# # Inference
# results = object_detection_model.infer(input_to_model)
# boxes, colors, names, labels = object_detection_model.postprocess_output(model_output=results, confidence_threshold=0.2)
# detections = draw_detections(boxes=boxes,colors=colors,names=names,img=rgb_img)
# save_image(img=detections, name='detection.jpg')

# #
# target_layers = [
#     object_detection_model.model.model.model[15]
# ]


# for _i,t in enumerate(target_layers):
#     cam = EigenCAM(model=object_detection_model.model, 
#                    target_layers=[t],
#                    reshape_transform=None,
#                    )
#     # Generate CAM for the image
#     #grayscale_cam = cam(input_to_model)
#     grayscale_cam = cam(tensor * 255)  # multiply by 255 because inside YOLO object of yolov8 the image is normalized
#     grayscale_cam = grayscale_cam[0, :, :]  
#     cam_image = show_cam_on_image(img_norm_np, grayscale_cam, use_rgb=True)
#     save_image(cam_image,f'detections_w_cam_s{_i}.jpg')

#     ################################
#     # REMOVE THE HEATMAP OUT OF THE BOUNDING BOXES
#     ################################
#     # Apply the renormalized CAM on the bounding boxes
#     renormalized_cam_image = renormalize_cam_in_bounding_boxes(boxes, colors, names, img_norm_np, grayscale_cam)
#     save_image(renormalized_cam_image,f'detections_w_cam_normalized_s{_i}.jpg')

#     # Concatenate the original image, CAM image, and renormalized CAM image for comparison
#     comparison_image = np.hstack((rgb_img, cam_image, renormalized_cam_image))
#     save_image(comparison_image,f'detections_s{_i}_all.jpg')

"""
[
/groups/tri110414/datasets/coco/images/val2017/000000041888.jpg,
/groups/tri110414/datasets/coco/images/val2017/000000555705.jpg,
/groups/tri110414/datasets/coco/images/val2017/000000500663.jpg,
/groups/tri110414/datasets/coco/images/val2017/000000418281.jpg,
/groups/tri110414/datasets/coco/images/val2017/000000025560.jpg,
/groups/tri110414/datasets/coco/images/val2017/000000185250.jpg,
/groups/tri110414/datasets/coco/images/val2017/000000572517.jpg,
/groups/tri110414/datasets/coco/images/val2017/000000516316.jpg,
/groups/tri110414/datasets/coco/images/val2017/000000382088.jpg,
/groups/tri110414/datasets/coco/images/val2017/000000476258.jpg,
/groups/tri110414/datasets/coco/images/val2017/000000515445.jpg,
/groups/tri110414/datasets/coco/images/val2017/000000173383.jpg,
/groups/tri110414/datasets/coco/images/val2017/000000063154.jpg,
/groups/tri110414/datasets/coco/images/val2017/000000551215.jpg,
/groups/tri110414/datasets/coco/images/val2017/000000544519.jpg,
/groups/tri110414/datasets/coco/images/val2017/000000213086.jpg
]
"""

