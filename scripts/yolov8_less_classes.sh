#!/bin/bash

# Generate a filename with the current timestamp
timestamp=$(date +%Y%m%d-%H%M%S)

## TAO COCO ##
### Pretained ###
nohup python custom_training.py -e 150 --model x --devices 1 --dataset tao_coco_5_classes_w_imgs_without_ann --batch_size 16 -cl_ms 5 --workers 10 > logs/{$timestamp}_nohup_yolov8x_5_cls_imgs_wout_ann.log 2>&1 &
# nohup python custom_training.py -e 150 --model x --devices 2 --dataset tao_coco_10_classes_w_imgs_without_ann --batch_size 16 -cl_ms 5 --workers 10 > logs/{$timestamp}_nohup_yolov8x_10_cls_imgs_wout_ann.log 2>&1 &
# nohup python custom_training.py -e 150 --model x --devices 3 --dataset tao_coco_20_classes_w_imgs_without_ann --batch_size 16 -cl_ms 5 --workers 10 > logs/{$timestamp}_nohup_yolov8x_20_cls_imgs_wout_ann.log 2>&1 &

nohup python custom_training.py -e 150 --model x --devices 2 --dataset tao_coco_5_classes --batch_size 16 -cl_ms 5 --workers 10 > logs/{$timestamp}_nohup_yolov8x_5_cls.log 2>&1 &
nohup python custom_training.py -e 150 --model x --devices 3 --dataset tao_coco_10_classes --batch_size 16 -cl_ms 5 --workers 10 --from_scratch > logs/{$timestamp}_nohup_yolov8x_10_cls.log 2>&1 &
# nohup python custom_training.py -e 150 --model x --devices 3 --dataset tao_coco_20_classes --batch_size 16 -cl_ms 5 --workers 10 > logs/{$timestamp}_nohup_yolov8x_20_cls.log 2>&1 &

### Scratch ###
nohup python custom_training.py -e 300 --model n --devices 1 --dataset tao_coco_5_classes --batch_size 16 -cl_ms 5 --workers 10 --from_scratch --val_every 5 > logs/{$timestamp}_nohup_yolov8n_5_cls_scratch.log 2>&1 &
nohup python custom_training.py -e 300 --model l --devices 3 --dataset tao_coco_5_classes --batch_size 16 -cl_ms 5 --workers 10 --from_scratch --val_every 5 > logs/{$timestamp}_nohup_yolov8l_5_cls_scratch.log 2>&1 &

### Finetune ###
nohup python custom_training.py -e 50 --lr 0.0005 --lrf 0.1 --model_path runs_COCO/20240306_1407_coco_5_classes_yolov8n_from_scratch/weights/best.pt --devices 3 --dataset tao_coco_5_classes --batch_size 16 -cl_ms 5 --workers 10 --val_every 5 > logs/{$timestamp}_nohup_yolov8n_5_cls_finetune.log 2>&1 &

## COCO ##
### Scratch ###
nohup python custom_training.py -e 150 --model n --devices 0 --dataset coco_5_classes --batch_size 16 -cl_ms 5 --workers 12 --val_every 5 --from_scratch > logs/{$timestamp}_nohup_yolov8n_COCO_5_cls_scratch.log 2>&1 &
nohup python custom_training.py -e 150 --model l --devices 2 --dataset coco_5_classes --batch_size 16 -cl_ms 5 --workers 12 --val_every 5 --from_scratch > logs/{$timestamp}_nohup_yolov8x_COCO_5_cls_scratch.log 2>&1 &
nohup python custom_training.py -e 150 --model x --devices 1 --dataset coco_5_classes --batch_size 16 -cl_ms 5 --workers 12 --val_every 5 --from_scratch > logs/{$timestamp}_nohup_yolov8x_COCO_5_cls_scratch.log 2>&1 &

## Validation ##
python custom_training.py --dataset tao_coco_5_classes --val_only --model_path runs_COCO/20240306_1407_coco_5_classes_yolov8n_from_scratch/weights/best.pt --devices 0 --batch_size 16 --workers 10
python custom_training.py --dataset tao_coco_5_classes --val_only --model_path runs_TAO/20240307_1426_tao_coco_5_classes_yolov8_finetuned/weights/best.pt --devices 0 --batch_size 16 --workers 10
python custom_training.py --dataset tao_coco_5_classes --val_only --model_path runs_COCO/20240306_1406_coco_5_classes_yolov8x_from_scratch/weights/best.pt --devices 2 --batch_size 16 --workers 10