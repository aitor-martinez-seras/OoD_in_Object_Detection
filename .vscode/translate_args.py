# Convert lauch.json to args
# ------------------------------------------------- #
args_list = [
                "--ood_method", "L1Ivis", "--which_internal_activations", "ftmaps_and_strides",
                "--cluster_optimization_metric", "silhouette",
                "--device", "0",
                "--benchmark", "cluster_methods",
                "--ood_datasets", "coco_ood", "coco_mixed", "owod", "--ood_split", "val", "--owod_task_ood", "t1",
                "--ind_info_creation_option", "valid_preds_one_stride",
                "--which_split", "train_val",
                "--load_ind_activations", 
                "--conf_thr_train", "0.15", "--conf_thr_test", "0.15", "--tpr_thr", "0.95",
                "--model_path", "runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt",
                "--ind_dataset", "owod", "--ind_split", "train", "--owod_task_ind", "t1",]

# ------------------------------------------------- #
# Transform the list of strings into a single string
only_string = " ".join(args_list)
print("-"*50)
print(only_string)
print("-"*50)
print()

# ************************************************************************* #

# Convert args to launch json
# Transform the string into a list of strings and print with double quotes
# ------------------------------------------------- #
string_with_args = "--which_split train_val --ood_method Cosine_cl_stride  --cluster_method HDBSCAN --benchmark unk_loc_enhancement --load_ind_activations --ood_datasets coco_ood coco_mixed --ood_method Cosine_cl_stride --which_internal_activations ftmaps_and_strides --cluster_optimization_metric silhouette --device 0 --ind_info_creation_option valid_preds_one_stride  --conf_thr_train 0.15 --conf_thr_test 0.15 --tpr_thr 0.95 --model_path runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt --ind_dataset owod --ind_split train --owod_task_ind t1 --ood_split val --owod_task_ood t1 > logs/benchmark_UNK.log 2>&1 &"
# ------------------------------------------------- #
args_list = string_with_args.split(" ")
# Now print the list with double quotes and commas between each element
print("-"*50)
print('"' + '", "'.join(args_list) + '"')
print("-"*50)