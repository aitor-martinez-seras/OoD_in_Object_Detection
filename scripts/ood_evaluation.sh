#!/bin/bash

# Generate a filename with the current timestamp
timestamp=$(date +%Y%m%d-%H%M%S)

nohup python ood_evaluation.py --conf_thr 0.02 --tpr_thr 0.95 --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_dataset owod --ood_split val --owod_task_ood all_task_test --device 0 --ood_method Energy --visualize_oods --model_path runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt  > logs/{$timestamp}_nohup_ood_eval_ENERGY.log 2>&1 &
nohup python ood_evaluation.py --conf_thr 0.02 --tpr_thr 0.95 --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_dataset owod --ood_split val --owod_task_ood all_task_test --device 0 --ood_method MSP --visualize_oods --model_path runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt > logs/{$timestamp}_nohup_ood_eval_MSP.log 2>&1 &
nohup python ood_evaluation.py --conf_thr 0.02 --tpr_thr 0.95 --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_dataset owod --ood_split val --owod_task_ood all_task_test --device 3 --ood_method L1_cl_stride --visualize_oods --model_path runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt > logs/{$timestamp}_nohup_ood_eval_L1.log 2>&1 &
nohup python ood_evaluation.py --conf_thr 0.02 --tpr_thr 0.95 --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_dataset owod --ood_split val --owod_task_ood all_task_test --device 3 --ood_method L2_cl_stride --visualize_oods --model_path runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt > logs/{$timestamp}_nohup_ood_eval_L1.log 2>&1 &

nohup python ood_evaluation.py --conf_thr 0.15 --tpr_thr 0.95 --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_dataset owod --ood_split val --owod_task_ood all_task_test --device 2 --ood_method L1_cl_stride --visualize_oods --model_path runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt 

python ood_evaluation.py --load_ind_activations --conf_thr 0.01  --tpr_thr 0.95 --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_dataset owod --ood_split val --owod_task_ood all_task_test --device 3 --ood_method L1_cl_stride --visualize_oods --model_path runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt
python ood_evaluation.py --conf_thr 0.0001  --tpr_thr 0.95 --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_dataset owod --ood_split val --owod_task_ood all_task_test --device 3 --ood_method L2_cl_stride --visualize_oods --model_path runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt
python ood_evaluation.py --conf_thr 0.001  --tpr_thr 0.95 --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_dataset owod --ood_split val --owod_task_ood all_task_test --device 3 --ood_method L1_cl_stride --visualize_oods --model_path runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt

