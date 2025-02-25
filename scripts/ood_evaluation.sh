#!/bin/bash

# Generate a filename with the current timestamp
timestamp=$(date +%Y%m%d-%H%M%S)

nohup python ood_evaluation.py --conf_thr 0.02 --tpr_thr 0.95 --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_dataset owod --ood_split val --owod_task_ood all_task_test --device 0 --ood_method Energy --visualize_oods --model_path runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt  > logs/{$timestamp}_nohup_ood_eval_ENERGY.log 2>&1 &
nohup python ood_evaluation.py --conf_thr 0.02 --tpr_thr 0.95 --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_dataset owod --ood_split val --owod_task_ood all_task_test --device 0 --ood_method MSP --visualize_oods --model_path runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt > logs/{$timestamp}_nohup_ood_eval_MSP.log 2>&1 &
nohup python ood_evaluation.py --conf_thr 0.02 --tpr_thr 0.95 --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_dataset owod --ood_split val --owod_task_ood all_task_test --device 3 --ood_method L1_cl_stride --visualize_oods --model_path runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt > logs/{$timestamp}_nohup_ood_eval_L1.log 2>&1 &
nohup python ood_evaluation.py --conf_thr 0.02 --tpr_thr 0.95 --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_dataset owod --ood_split val --owod_task_ood all_task_test --device 3 --ood_method L2_cl_stride --visualize_oods --model_path runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt > logs/{$timestamp}_nohup_ood_eval_L1.log 2>&1 &

nohup python ood_evaluation.py --conf_thr 0.15 --tpr_thr 0.95 --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_dataset owod --ood_split val --owod_task_ood all_task_test --device 2 --ood_method L1_cl_stride --visualize_oods --model_path runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt 

nohup python ood_evaluation.py --load_ind_activations --benchmark_conf --enhanced_unk_localization --ood_method Cosine_cl_stride --compute_metrics --device 2 --ind_info_creation_option valid_preds_one_stride --which_internal_activations ftmaps_and_strides --conf_thr 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_dataset coco_mixed --ood_split val --owod_task_ood all_task_test > logs/nohup_ood_benchmark.log 2>&1 &
nohup python ood_evaluation.py --load_ind_activations --enhanced_unk_localization --ood_method Cosine_cl_stride --compute_metrics --device 0 --ind_info_creation_option valid_preds_one_stride --which_internal_activations ftmaps_and_strides --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_dataset coco_mixed --ood_split val --owod_task_ood all_task_test

python ood_evaluation.py --load_ind_activations --conf_thr 0.01  --tpr_thr 0.95 --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_dataset owod --ood_split val --owod_task_ood all_task_test --device 3 --ood_method L1_cl_stride --visualize_oods --model_path runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt
python ood_evaluation.py --conf_thr 0.0001  --tpr_thr 0.95 --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_dataset owod --ood_split val --owod_task_ood all_task_test --device 3 --ood_method L2_cl_stride --visualize_oods --model_path runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt
python ood_evaluation.py --conf_thr 0.001  --tpr_thr 0.95 --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_dataset owod --ood_split val --owod_task_ood all_task_test --device 3 --ood_method L1_cl_stride --visualize_oods --model_path runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt


python ood_evaluation.py --ood_dataset coco_mixed --ood_method Cosine_cl_stride --cluster_method HDBSCAN --load_ind_activations --enhanced_unk_localization --compute_metrics --device 0 --ind_info_creation_option valid_preds_one_stride --which_internal_activations ftmaps_and_strides --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1  --ood_split val --owod_task_ood all_task_test

# Benchmarks
# Used TPR

# Conf threshold train

# Conf threshold test

# Clusters


# For testing unknown localization
nohup python ood_evaluation.py --enhanced_unk_localization --ood_method Cosine_cl_stride --cluster_method HDBSCAN --device 0 --compute_metrics --ind_info_creation_option valid_preds_one_stride --load_ind_activations --which_split train_val --which_internal_activations ftmaps_and_strides --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_dataset coco_mixed coco_ood --ood_split val --owod_task_ood all_task_test &
nohup python ood_evaluation.py --load_clusters --enhanced_unk_localization --ood_method Cosine_cl_stride --cluster_method HDBSCAN --device 0 --compute_metrics --ind_info_creation_option valid_preds_one_stride --load_ind_activations --which_split train_val --which_internal_activations ftmaps_and_strides --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_dataset coco_mixed coco_ood --ood_split val --owod_task_ood all_task_test &

# Fusion
nohup python ood_evaluation.py --compute_metrics --ood_method fusion-MSP-Energy-Sigmoid --which_internal_activations logits --which_split train_val --fusion_strategy score --cluster_optimization_metric silhouette --device 0 --ind_info_creation_option valid_preds_one_stride --load_ind_activations --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_datasets coco_ood coco_mixed --ood_split val --owod_task_ood all_task_test > logs/nohup_TripleFusion.log 2>&1 &
nohup python ood_evaluation.py --compute_metrics --ood_method fusion-CosineIvis-Cosine_cl_stride-Cosine_cl_stride --which_internal_activations ftmaps_and_strides --cluster_method KMeans_10-HDBSCAN-KMeans_10 --which_split train_val --fusion_strategy score --load_ind_activations  --cluster_optimization_metric silhouette --device 0 --ind_info_creation_option valid_preds_one_stride --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_datasets coco_ood coco_mixed --ood_split val --owod_task_ood all_task_test > logs/nohup_TripleFusion.log 2>&1 &