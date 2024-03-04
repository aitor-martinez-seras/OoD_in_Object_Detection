# nohup python custom_training_tao.py -e 150 -m m --devices 0 --dataset tao_coco_5_classes_w_imgs_without_ann --batch_size 16 -cl_ms 30 --workers 10 > nohup_yolov8n_5_cls.log &
# nohup python custom_training_tao.py -e 150 -m m --devices 1 --dataset tao_coco_10_classes_w_imgs_without_ann --batch_size 16 -cl_ms 30 --workers 10 > nohup_yolov8n_10_cls.log &
# nohup python custom_training_tao.py -e 150 -m m --devices 2 --dataset tao_coco_20_classes_w_imgs_without_ann --batch_size 16 -cl_ms 30 --workers 10 > nohup_yolov8n_20_cls.log &

nohup python custom_training_tao.py -e 150 --model x --devices 1 --dataset tao_coco_5_classes_w_imgs_without_ann --batch_size 16 -cl_ms 5 --workers 10 > nohup_yolov8x_5_cls_imgs_wout_ann.log &
# nohup python custom_training_tao.py -e 150 --model x --devices 2 --dataset tao_coco_10_classes_w_imgs_without_ann --batch_size 16 -cl_ms 5 --workers 10 > nohup_yolov8x_10_cls_imgs_wout_ann.log &
# nohup python custom_training_tao.py -e 150 --model x --devices 3 --dataset tao_coco_20_classes_w_imgs_without_ann --batch_size 16 -cl_ms 5 --workers 10 > nohup_yolov8x_20_cls_imgs_wout_ann.log &

nohup python custom_training_tao.py -e 150 --model x --devices 2 --dataset tao_coco_5_classes --batch_size 16 -cl_ms 5 --workers 10 > nohup_yolov8x_5_cls.log &
nohup python custom_training_tao.py -e 150 --model x --devices 3 --dataset tao_coco_10_classes --batch_size 16 -cl_ms 5 --workers 10 > nohup_yolov8x_10_cls.log &
# nohup python custom_training_tao.py -e 150 --model x --devices 3 --dataset tao_coco_20_classes --batch_size 16 -cl_ms 5 --workers 10 > nohup_yolov8x_20_cls.log &