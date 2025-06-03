## No Method, for the VOC test performance results with RAW model
python ood_evaluation.py --benchmark conf_thr_test --ood_method NoMethod --cluster_method one --which_internal_activations logits --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_NoMethod.log 2>&1 &

## RQ1
# Vanilla runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Block 1: RQ1 - L1 Methods with different cluster strategies"
python ood_evaluation.py --benchmark conf_thr_test --ood_method L1_cl_stride --cluster_method one --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_L1_one.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --ood_method L1_cl_stride --cluster_method KMeans --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_L1_KMeans.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --ood_method L1_cl_stride --cluster_method KMeans_10 --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_L1_KMeans_10.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --ood_method L1_cl_stride --cluster_method HDBSCAN --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_L1_HDBSCAN.log 2>&1 &
pids+=($!)

for pid in "${pids[@]}"; do
    wait "$pid"
done


echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Block 2: RQ1 - L2 Methods with different cluster strategies"
python ood_evaluation.py --benchmark conf_thr_test --ood_method L2_cl_stride --cluster_method one --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_L2_one.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --ood_method L2_cl_stride --cluster_method KMeans --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_L2_KMeans.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --ood_method L2_cl_stride --cluster_method KMeans_10 --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_L2_KMeans_10.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --ood_method L2_cl_stride --cluster_method HDBSCAN --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_L2_HDBSCAN.log 2>&1 &
pids+=($!)

for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Block 3: RQ1 - L2 Methods with different cluster strategies"
python ood_evaluation.py --benchmark conf_thr_test --ood_method Cosine_cl_stride --cluster_method one --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_Cosine_one.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --ood_method Cosine_cl_stride --cluster_method KMeans --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_Cosine_KMeans.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --ood_method Cosine_cl_stride --cluster_method KMeans_10 --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_Cosine_KMeans_10.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --ood_method Cosine_cl_stride --cluster_method HDBSCAN --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_Cosine_HDBSCAN.log 2>&1 &
pids+=($!)

for pid in "${pids[@]}"; do
    wait "$pid"
done

# SDR
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Block 4"
python ood_evaluation.py --benchmark conf_thr_test --ood_method L1Ivis --cluster_method one --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_L1Ivis_One.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --ood_method L1Ivis --cluster_method KMeans --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_L1Ivis_KMeans.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --ood_method L1Ivis --cluster_method KMeans_10 --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_L1Ivis_KMeans_10.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --ood_method L1Ivis --cluster_method HDBSCAN --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_L1Ivis_HDBSCAN.log 2>&1 &
pids+=($!)

for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Block 5"

pids=()
python ood_evaluation.py --benchmark conf_thr_test --ood_method L2Ivis --cluster_method one --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_L2Ivis_One.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --ood_method L2Ivis --cluster_method KMeans --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_L2Ivis_KMeans.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --ood_method L2Ivis --cluster_method KMeans_10 --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_L2Ivis_KMeans_10.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --ood_method L2Ivis --cluster_method HDBSCAN --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_L2Ivis_HDBSCAN.log 2>&1 &
pids+=($!)

for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Block 6"

pids=()
python ood_evaluation.py --benchmark conf_thr_test --ood_method CosineIvis --cluster_method one --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_CosineIvis_One.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --ood_method CosineIvis --cluster_method KMeans --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_CosineIvis_KMeans.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --ood_method CosineIvis --cluster_method KMeans_10 --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_CosineIvis_KMeans_10.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --ood_method CosineIvis --cluster_method HDBSCAN --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_CosineIvis_HDBSCAN.log 2>&1 &
pids+=($!)

for pid in "${pids[@]}"; do
    wait "$pid"
done

# EUL
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Block 7"

pids=()
python ood_evaluation.py --enhanced_unk_localization --benchmark conf_thr_test --ood_method L1_cl_stride --cluster_method one --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_EUL_L1_one.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --enhanced_unk_localization --benchmark conf_thr_test --ood_method L1_cl_stride --cluster_method KMeans --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_EUL_L1_KMeans.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --enhanced_unk_localization --benchmark conf_thr_test --ood_method L1_cl_stride --cluster_method KMeans_10 --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_EUL_L1_KMeans_10.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --enhanced_unk_localization --benchmark conf_thr_test --ood_method L1_cl_stride --cluster_method HDBSCAN --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_EUL_L1_HDBSCAN.log 2>&1 &
pids+=($!)

for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Block 8"

pids=()
python ood_evaluation.py --enhanced_unk_localization --benchmark conf_thr_test --ood_method L2_cl_stride --cluster_method one --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_EUL_L2_one.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --enhanced_unk_localization --benchmark conf_thr_test --ood_method L2_cl_stride --cluster_method KMeans --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_EUL_L2_KMeans.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --enhanced_unk_localization --benchmark conf_thr_test --ood_method L2_cl_stride --cluster_method KMeans_10 --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_EUL_L2_KMeans_10.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --enhanced_unk_localization --benchmark conf_thr_test --ood_method L2_cl_stride --cluster_method HDBSCAN --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_EUL_L2_HDBSCAN.log 2>&1 &
pids+=($!)

for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Block 9"

pids=()
python ood_evaluation.py --enhanced_unk_localization --benchmark conf_thr_test --ood_method Cosine_cl_stride --cluster_method one --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_EUL_Cosine_one.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --enhanced_unk_localization --benchmark conf_thr_test --ood_method Cosine_cl_stride --cluster_method KMeans --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_EUL_Cosine_KMeans.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --enhanced_unk_localization --benchmark conf_thr_test --ood_method Cosine_cl_stride --cluster_method KMeans_10 --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_EUL_Cosine_KMeans_10.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --enhanced_unk_localization --benchmark conf_thr_test --ood_method Cosine_cl_stride --cluster_method HDBSCAN --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_EUL_Cosine_HDBSCAN.log 2>&1 &
pids+=($!)

for pid in "${pids[@]}"; do
    wait "$pid"
done

## RQ2
# Logits
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Block 10"

pids=()
python ood_evaluation.py --benchmark conf_thr_test --ood_method MSP --cluster_method one --which_internal_activations logits --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_MSP.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --ood_method Energy --cluster_method one --which_internal_activations logits --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_Energy.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --ood_method ODIN --cluster_method one --which_internal_activations logits --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_ODIN.log 2>&1 &
pids+=($!)

for pid in "${pids[@]}"; do
    wait "$pid"
done

## Fusion
# or
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Block 11"

pids=()
python ood_evaluation.py --benchmark conf_thr_test --fusion_strategy or --ood_method fusion-MSP-Energy --cluster_method one --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_or_MSP-Energy.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --fusion_strategy or --ood_method fusion-MSP-ODIN --cluster_method one --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_or_MSP-ODIN.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --fusion_strategy or --ood_method fusion-MSP-Cosine_cl_stride --cluster_method HDBSCAN --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_or_MSP-CosineHDBSCAN.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --fusion_strategy or --ood_method fusion-MSP-CosineIvis --cluster_method HDBSCAN --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_or_MSP-CosineIvisHDBSCAN.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --fusion_strategy or --ood_method fusion-Cosine_cl_stride-CosineIvis --cluster_method KMeans_10-HDBSCAN --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_or_CosineKMeans10-CosineIvisHDBSCAN.log 2>&1 &
pids+=($!)

for pid in "${pids[@]}"; do
    wait "$pid"
done

#
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Block 12"

pids=()
python ood_evaluation.py --benchmark conf_thr_test --fusion_strategy or --ood_method fusion-MSP-L1_cl_stride --cluster_method one --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_or_MSP-L1one.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --fusion_strategy or --ood_method fusion-MSP-L1Ivis --cluster_method one --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_or_MSP-L1Ivisone.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --fusion_strategy or --ood_method fusion-L1_cl_stride-CosineIvis --cluster_method one-HDBSCAN --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_or_L1one-CosineIvisHDBSCAN.log 2>&1 &
pids+=($!)

for pid in "${pids[@]}"; do
    wait "$pid"
done

# and
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Block 13"

pids=()
python ood_evaluation.py --benchmark conf_thr_test --fusion_strategy and --ood_method fusion-MSP-Energy --cluster_method one --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_and_MSP-Energy.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --fusion_strategy and --ood_method fusion-MSP-ODIN --cluster_method one --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_and_MSP-ODIN.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --fusion_strategy and --ood_method fusion-MSP-Cosine_cl_stride --cluster_method HDBSCAN --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_and_MSP-CosineHDBSCAN.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --fusion_strategy and --ood_method fusion-MSP-CosineIvis --cluster_method HDBSCAN --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_and_MSP-CosineIvisHDBSCAN.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --fusion_strategy and --ood_method fusion-Cosine_cl_stride-CosineIvis --cluster_method KMeans_10-HDBSCAN --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_and_CosineKMeans10-CosineIvisHDBSCAN.log 2>&1 &
pids+=($!)

for pid in "${pids[@]}"; do
    wait "$pid"
done
#
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Block 14"

pids=()
python ood_evaluation.py --benchmark conf_thr_test --fusion_strategy and --ood_method fusion-MSP-L1_cl_stride --cluster_method one --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_and_MSP-L1one.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --fusion_strategy and --ood_method fusion-MSP-L1Ivis --cluster_method one --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_and_MSP-L1Ivisone.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --fusion_strategy and --ood_method fusion-L1_cl_stride-CosineIvis --cluster_method one-HDBSCAN --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_and_L1one-CosineIvisHDBSCAN.log 2>&1 &
pids+=($!)

for pid in "${pids[@]}"; do
    wait "$pid"
done

# score
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Block 15"

pids=()
python ood_evaluation.py --benchmark conf_thr_test --fusion_strategy score --ood_method fusion-MSP-Energy --cluster_method one --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_score_MSP-Energy.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --fusion_strategy score --ood_method fusion-MSP-ODIN --cluster_method one --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_score_MSP-ODIN.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --fusion_strategy score --ood_method fusion-MSP-Cosine_cl_stride --cluster_method HDBSCAN --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_score_MSP-CosineHDBSCAN.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --fusion_strategy score --ood_method fusion-MSP-CosineIvis --cluster_method HDBSCAN --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_score_MSP-CosineIvisHDBSCAN.log 2>&1 &
pids+=($!)
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --fusion_strategy score --ood_method fusion-Cosine_cl_stride-CosineIvis --cluster_method KMeans_10-HDBSCAN --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_score_CosineKMeans10-CosineIvisHDBSCAN.log 2>&1 &
pids+=($!)

for pid in "${pids[@]}"; do
    wait "$pid"
done
#
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Block 16"
python ood_evaluation.py --benchmark conf_thr_test --fusion_strategy score --ood_method fusion-MSP-L1_cl_stride --cluster_method one --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_score_MSP-L1one.log 2>&1 &
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --fusion_strategy score --ood_method fusion-MSP-L1Ivis --cluster_method one --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_score_MSP-L1Ivisone.log 2>&1 &
sleep 90
python ood_evaluation.py --benchmark conf_thr_test --fusion_strategy score --ood_method fusion-L1_cl_stride-CosineIvis --cluster_method one-HDBSCAN --which_internal_activations ftmaps_and_strides --which_split train_val  --load_ind_activations --ood_datasets owod coco_ood coco_mixed --device 2 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_V10_conf_trest_score_L1one-CosineIvisHDBSCAN.log 2>&1 &
