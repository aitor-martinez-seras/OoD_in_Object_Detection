from pathlib import Path
from datetime import datetime
from typing import Literal, List
import os

from tap import Tap

import numpy as np
from ultralytics import YOLO
from object_detection_CAM_xai.xai.common.object_detection_models import YOLOv8CAM
from object_detection_CAM_xai.xai.common._utils import process_image, draw_detections, save_image, renormalize_cam_in_bounding_boxes, letterbox

from pytorch_grad_cam import EigenCAM, AblationCAM, ScoreCAM,GradCAM  # Grad-CAM implementation
from pytorch_grad_cam.utils.image import show_cam_on_image


# GPU
gpu_number = str(2)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_number
device = 'cuda:0'

# Set model
model = YOLO("runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt", task='detect')
object_detection_model = YOLOv8CAM(model, device=device)

# Target layers. Are passed one by one
target_layers_numbers = [15, 18, 21] 
target_layers = [object_detection_model.model.model.model[n] for n in target_layers_numbers]

# Load images
if False:
    path = "/groups/tri110414/datasets/coco/images/val2017/"
    image_name = "000000041888.jpg"
    tensor, rgb_img, img_norm_np = process_image(path+image_name)
    input_to_model = rgb_img
else:
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

PREFIX = 'detections'

FOLDER_PATH = Path('al_expl_imgs')
FOLDER_PATH.mkdir(exist_ok=True)

for _img_idx, img_path in enumerate(img_paths):
    tensor, rgb_img, img_norm_np = process_image(img_path, use_letterbox=True)
    input_to_model = rgb_img
    # Inference
    results = object_detection_model.infer(input_to_model)
    boxes, colors, names, labels = object_detection_model.postprocess_output(model_output=results, confidence_threshold=0.2)
    detections = draw_detections(boxes=boxes,colors=colors,names=names,img=rgb_img)
    save_image(img=detections, name='detection.jpg')

    for _i,t in enumerate(target_layers):
        cam = EigenCAM(model=object_detection_model.model, 
                    target_layers=[t],
                    reshape_transform=None,
                    )
        # Generate CAM for the image
        #grayscale_cam = cam(input_to_model)
        grayscale_cam = cam(tensor * 255)  # multiply by 255 because inside YOLO object of yolov8 the image is normalized
        grayscale_cam = grayscale_cam[0, :, :]  
        cam_image = show_cam_on_image(img_norm_np, grayscale_cam, use_rgb=True)
        #save_image(cam_image, (FOLDER_PATH / f'{PREFIX}_img{_img_idx:02d}_w_cam_s_layer{target_layers_numbers[_i]}.png').as_posix())

        ################################
        # REMOVE THE HEATMAP OUT OF THE BOUNDING BOXES
        ################################
        # Apply the renormalized CAM on the bounding boxes
        renormalized_cam_image = renormalize_cam_in_bounding_boxes(boxes, colors, names, img_norm_np, grayscale_cam)
        #save_image(renormalized_cam_image, (FOLDER_PATH / f'{PREFIX}_img{_img_idx:02d}_w_cam_normalized_s_layer{target_layers_numbers[_i]}.png').as_posix())

        # Concatenate the original image, CAM image, and renormalized CAM image for comparison
        comparison_image = np.hstack((rgb_img, cam_image, renormalized_cam_image))
        save_image(comparison_image, (FOLDER_PATH / f'{PREFIX}_img{_img_idx:02d}_s_layer{target_layers_numbers[_i]}_all.png').as_posix())
    
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

