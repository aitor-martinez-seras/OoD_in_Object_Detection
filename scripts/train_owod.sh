#!/bin/bash

# Generate a filename with the current timestamp
timestamp=$(date +%Y%m%d-%H%M%S)

## OWOD ##
### Scratch ###
# Task 1
nohup python custom_training.py -e 400 --lr 0.001 --lrf 0.1 --owod_task t1 --model yolov8 --model_size s --devices 1 --dataset owod --batch_size 16 -cl_ms 5 --workers 15 --val_every 5 --from_scratch > logs/{$timestamp}_nohup_yolov8n_OWOD_t1_scratch.log 2>&1 &
nohup python custom_training.py -e 300 --lr 0.005 --lrf 0.05 --owod_task t1 --model yolov8 --model_size l --devices 2 --dataset owod --batch_size 16 -cl_ms 5 --workers 12 --val_every 5 --from_scratch > logs/{$timestamp}_nohup_yolov8l_OWOD_t1_scratch_lr0005.log 2>&1 &
nohup python custom_training.py -e 300 --lr 0.001 --lrf 0.1 --owod_task t1 --model yolov8 --model_size l --devices 3 --dataset owod --batch_size 16 -cl_ms 5 --workers 12 --val_every 5 --from_scratch > logs/{$timestamp}_nohup_yolov8l_OWOD_t1_scratch_lr0001.log 2>&1 &
nohup python custom_training.py -e 300 --lr 0.0005 --lrf 0.1 --owod_task t1 --model yolov8 --model_size l --devices 2 --dataset owod --batch_size 16 -cl_ms 5 --workers 12 --val_every 5 --from_scratch > logs/{$timestamp}_nohup_yolov8l_OWOD_t1_scratch_lr00005.log 2>&1 &
nohup python custom_training.py -e 300 --lr 0.001 --lrf 0.1 --cos_lr --owod_task t1 --model yolov8 --model_size l --devices 1 --dataset owod --batch_size 16 -cl_ms 5 --workers 12 --val_every 5 --from_scratch > logs/{$timestamp}_nohup_yolov8l_OWOD_t1_scratch_lr0001_cosineLR.log 2>&1 &
nohup python custom_training.py -e 300 --lr 0.0005 --lrf 0.2 --owod_task t1 --model yolov8 --model_size l --devices 2 --dataset owod --batch_size 16 -cl_ms 1 --workers 12 --val_every 5 --from_scratch > logs/{$timestamp}_nohup_yolov8l_OWOD_t1_scratch_lr00005_lf02.log 2>&1 &
nohup python custom_training.py -e 300 --lr 0.001 --lrf 0.15 --owod_task t1 --model yolov8 --model_size l --devices 3 --dataset owod --batch_size 16 -cl_ms 5 --workers 12 --val_every 5 --from_scratch > logs/{$timestamp}_nohup_yolov8l_OWOD_t1_scratch_lr0001_lf015.log 2>&1 &

nohup python custom_training.py -e 300 --lr 0.001 --lrf 0.15 --owod_task t1 --model yolov5 --model_size l --devices 3 --dataset owod --batch_size 16 -cl_ms 5 --workers 12 --val_every 5 --from_scratch > logs/{$timestamp}_nohup_yolov5l_OWOD_t1_scratch_lr0001_lf015.log 2>&1 &

nohup python custom_training.py --imagenet_pretrained_backbone -e 300 --lr 0.001 --lrf 0.15 --owod_task t1 --model yolov5 --model_size l --devices 3 --dataset owod --batch_size 16 -cl_ms 5 --workers 12 --val_every 5 --from_scratch > logs/{$timestamp}_nohup_yolov8l_OWOD_t1_imagenet_pretrained_lr0001_lf015.log 2>&1 &





## Validation ##
python custom_training.py --dataset tao_coco_5_classes --val_only --model_path runs_COCO/20240306_1407_coco_5_classes_yolov8n_from_scratch/weights/best.pt --devices 0 --batch_size 16 --workers 10
python custom_training.py --dataset tao_coco_5_classes --val_only --model_path runs_COCO/20240306_1406_coco_5_classes_yolov8x_from_scratch/weights/best.pt --devices 2 --batch_size 16 --workers 10