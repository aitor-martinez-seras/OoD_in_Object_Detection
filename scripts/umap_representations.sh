# M models
nohup python create_umap_representation.py --model_folder runs_TAO/20240229_1946_tao_coco_5_classes_w_imgs_wthout_ann_yolov8m_pretrained --number_of_known_classes 5 --one_umap_per_stride True --device 2 --grid_search_umap True > umap_grid_search_5_cls_M.log &

# X models pretrained
nohup python create_umap_representation.py --model_folder runs_TAO/20240301_1816_tao_coco_5_classes_yolov8x_pretrained --number_of_known_classes 5 --one_umap_per_stride True --device 2 --grid_search_umap True > umap_grid_search_5_cls.log &
nohup python create_umap_representation.py --model_folder runs_TAO/20240301_1816_tao_coco_10_classes_yolov8x_pretrained --number_of_known_classes 10 --one_umap_per_stride True --device 1 --grid_search_umap True > umap_grid_search_10_cls.log &

# X models SCRATCH
python create_umap_representation.py --model_folder runs_TAO/20240305_1150_tao_coco_5_classes_yolov8n_from_scratch --number_of_known_classes 5 --one_umap_per_stride True --device 1 --grid_search_umap False

### Models from scratch in COCO ###
python create_umap_representation.py --device 0 --model_folder runs_COCO/20240306_1407_coco_5_classes_yolov8n_from_scratch --dataset tao_coco --split train --number_of_known_classes 5 --one_umap_per_stride True  --grid_search_umap False
python create_umap_representation.py --device 0 --model_folder runs_COCO/20240306_1407_coco_5_classes_yolov8n_from_scratch --dataset coco --split val --number_of_known_classes 5 --one_umap_per_stride True --grid_search_umap False
python create_umap_representation.py --device 0 --model_folder runs_COCO/20240306_1406_coco_5_classes_yolov8x_from_scratch --dataset tao_coco --split train --number_of_known_classes 5 --one_umap_per_stride True --grid_search_umap False
python create_umap_representation.py --device 0 --model_folder runs_COCO/20240306_1406_coco_5_classes_yolov8x_from_scratch --dataset coco --split val --number_of_known_classes 5 --one_umap_per_stride True --grid_search_umap False
nohup python create_umap_representation.py --device 2 --model_folder runs_COCO/20240306_1406_coco_5_classes_yolov8x_from_scratch --dataset coco --split val --number_of_known_classes 5 --one_umap_per_stride True --grid_search_umap True > umap_grid_search_5_cls_COCO.log 2>&1 &
nohup python create_umap_representation.py --device 2 --model_folder runs_COCO/20240306_1406_coco_5_classes_yolov8x_from_scratch --dataset tao_coco --split train --number_of_known_classes 5 --one_umap_per_stride True --grid_search_umap True > umap_grid_search_5_cls_TAO_coco.log 2>&1 &

# Finetuned in TAO
python create_umap_representation.py --device 0 --model_folder runs_TAO/20240307_1426_tao_coco_5_classes_yolov8_finetuned --dataset tao_coco --split train --number_of_known_classes 5 --one_umap_per_stride True --grid_search_umap False
python create_umap_representation.py --device 0 --model_folder runs_TAO/20240307_1426_tao_coco_5_classes_yolov8_finetuned --dataset coco --split val --number_of_known_classes 5 --one_umap_per_stride True --grid_search_umap False