from pathlib import Path
from datetime import datetime
from typing import Literal, List
import os

from tap import Tap

import numpy as np
from ultralytics import YOLO
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

import cv2
import torch
import torchvision.transforms as transforms
import torchvision

import xml.etree.ElementTree as ET

from YOLOv8_Explainer.utils import letterbox

from yolo_drise.xai.drise import DRISE
from yolo_drise.utils.utils import tensor_imshow, get_class_name_coco

# add the following to avoid ssl issues from the server
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

print('---------------------------------')
print('D-RISE YOLOv8')
print('---------------------------------')

# GPU
gpu_number = str(3)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_number
device = 'cuda:0'

# Set model
model = YOLO("runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt", task='detect')
cls_names = model.names

# Load image
# path = "/groups/tri110414/datasets/coco/images/val2017/"
# image_name = "000000041888.jpg"
# img_path = "/groups/tri110414/datasets/coco/images/val2017/000000041888.jpg"
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
selected_image = 7
img_path = img_paths[7]

# preprocess image
img = np.array(Image.open(img_path))
img = letterbox(img, auto=False)[0]  # add padding to (640,640)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.float32(img) / 255.0  # type: ignore
tensor = (
    torch.from_numpy(np.transpose(img, axes=[2, 0, 1]))
    .unsqueeze(0)
    .to(device)
)

# Predict image to extract target bounding boxes
#results = model(img_path)[0]  # Asi el modelo carga la imagen sin letterbox
results = model(tensor*255)[0]  # multiply by 255 because inside YOLO object of yolov8 the image is normalized
preds_cls = results.boxes.cls.cpu()
preds_boxes = results.boxes.xyxy.cpu()

#########################
# Select the target class (or classes) and bounding box
#########################
target_bbox_idx = 0
target_classes = [int(preds_cls[target_bbox_idx].item())] + [14]
target_bbox = preds_boxes[target_bbox_idx].int().tolist()

# Plot the image with the bounding boxes
# from ultralytics.yolo.utils.ops import non_max_suppression
# from torchvision.utils import draw_bounding_boxes
# img_with_boxes = draw_bounding_boxes(
#     (tensor * 255).to(torch.uint8)[0],
#     preds_boxes,
#     [model.names[_cls.item()] for _cls in preds_cls],
#     colors=None,
#     width=2
# )
# plt.imshow(img_with_boxes.permute(1, 2, 0))
# plt.savefig('drise_imagen_prueba.png')

#########################
# Generate Explainer Instance
#########################
input_size = (640, 640)
gpu_batch = 64
number_of_masks = 6000
stride = 8
p1 = 0.5
explainer = DRISE(model=model, 
                  input_size=input_size, 
                  device=device,
                  gpu_batch=gpu_batch)

# Generate masks for RISE or use the saved ones.
generate_new = False
mask_file = f"./yolo_drise/masks/masks_640x640.npy"

if generate_new or not os.path.isfile(mask_file):
    # explainer.generate_masks(N=5000, s=8, p1=0.1, savepath= mask_file)
    explainer.generate_masks(N=number_of_masks, s=stride, p1=p1, savepath=mask_file)
else:
    explainer.load_masks(mask_file)
    print('Masks are loaded.')


#########################
# Explain & Visualize (option 2) --> no saliency map exports and visualize straight
#########################
# apply xai
saliency = explainer(x=tensor,
                     target_class_indices=target_classes,  # List[int]
                     target_bbox=target_bbox)  # List[int, int, int, int]

# # Scale saliency map to [0, 1]
# a, b = 0, 1
# saliency = (b-a) * (saliency - saliency.min())/(saliency.max() - saliency.min()) + a
# plt.imshow(saliency, cmap='jet')
# plt.colorbar()
# plt.savefig(f'A_drise.png')
# plt.close()


plt.figure(figsize=(10, 5 * len(target_classes)))
for i, cl in enumerate(target_classes):
    # Plot original image
    plt.subplot(len(target_classes), 2, 2*i + 1)
    plt.axis('off')
    plt.title(cls_names[cl])
    tensor_imshow(inp=tensor.squeeze(0))
    # plt.imshow(tensor.cpu().numpy().squeeze(0).transpose((1, 2, 0)))

    # Plot saliency map for the class
    plt.subplot(len(target_classes), 2, 2*i + 2)
    plt.axis('off')
    plt.title(cls_names[cl])
    tensor_imshow(inp=tensor.squeeze(0))
    # plt.imshow(tensor.cpu().numpy().squeeze(0).transpose((1, 2, 0)))
    plt.imshow(saliency[cl], cmap='jet', alpha=0.5)
    plt.colorbar(fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(f'yolo_drise/drise_{selected_image:2d}_{cls_names[cl]}.png')
#plt.show()
plt.close()
