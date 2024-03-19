### Models from scratch in COCO ###
# Grid search
nohup python create_umap_representation.py --device 2 --model_folder runs_COCO/20240306_1406_coco_5_classes_yolov8x_from_scratch --dataset coco --split val --number_of_known_classes 5 --one_umap_per_stride True --grid_search_umap True > umap_grid_search_5_cls_COCO.log 2>&1 &
nohup python create_umap_representation.py --device 2 --model_folder runs_COCO/20240306_1406_coco_5_classes_yolov8x_from_scratch --dataset tao_coco --split val --number_of_known_classes 5 --one_umap_per_stride True --grid_search_umap True > umap_grid_search_5_cls_TAO_coco.log 2>&1 &
# Direct
nohup python create_umap_representation.py --device 0 --model_folder runs_COCO/20240306_1406_coco_5_classes_yolov8x_from_scratch --number_of_known_classes 5 --n_neighbors 20 --metric euclidean --min_dist 0.1 --target_weight 0.2 --one_umap_per_stride True --grid_search_umap False --mode pca_umap --dataset coco --split val &
nohup python create_umap_representation.py --device 1 --model_folder runs_COCO/20240306_1406_coco_5_classes_yolov8x_from_scratch --number_of_known_classes 5 --n_neighbors 20 --metric euclidean --min_dist 0.1 --target_weight 0.2 --one_umap_per_stride True --grid_search_umap False --mode pca_umap --dataset tao_coco --split val &

nohup python create_umap_representation.py --device 0 --model_folder runs_COCO/20240306_1406_coco_5_classes_yolov8x_from_scratch --number_of_known_classes 5 --n_neighbors 20 --metric euclidean --min_dist 0.1 --target_weight 0.2 --one_umap_per_stride True --grid_search_umap False --mode pca --dataset coco --split val &
nohup python create_umap_representation.py --device 1 --model_folder runs_COCO/20240306_1406_coco_5_classes_yolov8x_from_scratch --number_of_known_classes 5 --n_neighbors 20 --metric euclidean --min_dist 0.1 --target_weight 0.2 --one_umap_per_stride True --grid_search_umap False --mode pca --dataset tao_coco --split val &

nohup python create_umap_representation.py --device 0 --model_folder runs_COCO/20240306_1406_coco_5_classes_yolov8x_from_scratch --number_of_known_classes 5 --n_neighbors 20 --metric euclidean --min_dist 0.1 --target_weight 0.2 --one_umap_per_stride True --grid_search_umap False --mode umap --dataset coco --split val &
nohup python create_umap_representation.py --device 1 --model_folder runs_COCO/20240306_1406_coco_5_classes_yolov8x_from_scratch --number_of_known_classes 5 --n_neighbors 20 --metric euclidean --min_dist 0.1 --target_weight 0.2 --one_umap_per_stride True --grid_search_umap False --mode umap --dataset tao_coco --split val &

# Finetuned in TAO
python create_umap_representation.py --device 0 --model_folder runs_TAO/20240307_1426_tao_coco_5_classes_yolov8_finetuned --dataset tao_coco --split val --number_of_known_classes 5 --one_umap_per_stride True --grid_search_umap False
python create_umap_representation.py --device 0 --model_folder runs_TAO/20240307_1426_tao_coco_5_classes_yolov8_finetuned --dataset coco --split val --number_of_known_classes 5 --one_umap_per_stride True --grid_search_umap False