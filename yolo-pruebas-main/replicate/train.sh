nohup python custom_training.py --model custom_yolov8 -e 300 --lr 0.0005 --lrf 0.2 --owod_task t1 --model_size l --devices 2 --dataset owod --batch_size 16 -cl_ms 1 --workers 12 --val_every 5 --from_scratch > logs/nohup_custom_yolov8l_OWOD_t1_scratch_lr00005_lf02.log 2>&1 &