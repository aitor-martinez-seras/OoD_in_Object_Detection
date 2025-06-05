# V8  runs_OWOD/20240313_1407_owod_t1_yolov8l_from_scratch/weights/best.pt - > --lr 0.001 --lrf 0.15
nohup python custom_training.py --model yolov8 -e 300 --lr 0.001 --lrf 0.15 --owod_task t1 --model_size l --devices 3 --dataset owod --batch_size 16 -cl_ms 10 --workers 12 --val_every 5 --from_scratch > logs/nohup_custom_yolov8l_OWOD_t1_scratch_lr0001_lf015.log 2>&1 &
#nohup python custom_training.py --model yolov8 -e 300 --lr 0.0005 --lrf 0.2 --owod_task t1 --model_size l --devices 2 --dataset owod --batch_size 16 -cl_ms 10 --workers 12 --val_every 5 --from_scratch > logs/nohup_custom_yolov8l_OWOD_t1_scratch_lr00005_lf02.log 2>&1 &
# Medium
nohup python custom_training.py --model yolov8 -e 300 --lr 0.001 --lrf 0.15 --owod_task t1 --model_size m --devices 1 --dataset owod --batch_size 16 -cl_ms 10 --workers 12 --val_every 5 --from_scratch > logs/nohup_custom_yolov8m_OWOD_t1_scratch_lr0001_lf015.log 2>&1 &

# V9
# Best results
nohup python custom_training.py --model yolov9 -e 300 --lr 0.0003 --lrf 0.2 --owod_task t1 --model_size m --devices 0 --dataset owod --batch_size 16 -cl_ms 10 --workers 12 --val_every 5 --from_scratch > logs/nohup_custom_yolov9l_OWOD_t1_scratch_lr00003_lf02.log 2>&1 &
#
nohup python custom_training.py --model yolov9 -e 300 --lr 0.0001 --lrf 0.2 --owod_task t1 --model_size m --devices 0 --dataset owod --batch_size 16 -cl_ms 10 --workers 12 --val_every 5 --from_scratch > logs/nohup_custom_yolov9l_OWOD_t1_scratch_lr00001_lf02.log 2>&1 &

# V10 
# Best results  runs_OWOD/20250525_1733_owod_t1_yolov10l_from_scratch/weights/best.pt
nohup python custom_training.py --model yolov10 -e 300 --lr 0.0003 --lrf 0.2 --owod_task t1 --model_size l --devices 0 --dataset owod --batch_size 16 -cl_ms 10 --workers 12 --val_every 5 --from_scratch > logs/nohup_custom_yolov10l_OWOD_t1_scratch_lr00003_lf02.log 2>&1 &
#
nohup python custom_training.py --model yolov10 -e 300 --lr 0.0001 --lrf 0.2 --owod_task t1 --model_size l --devices 0 --dataset owod --batch_size 16 -cl_ms 10 --workers 12 --val_every 5 --from_scratch > logs/nohup_custom_yolov10l_OWOD_t1_scratch_lr00001_lf02.log 2>&1 &
nohup python custom_training.py --model yolov10 -e 300 --lr 0.0003 --lrf 0.2 --owod_task t1 --model_size l --devices 0 --dataset owod --batch_size 16 -cl_ms 10 --workers 12 --val_every 5 --from_scratch > logs/nohup_custom_yolov10l_OWOD_t1_scratch_lr00003_lf02.log 2>&1 &
nohup python custom_training.py --model yolov10 -e 300 --lr 0.0005 --lrf 0.2 --owod_task t1 --model_size l --devices 0 --dataset owod --batch_size 16 -cl_ms 10 --workers 12 --val_every 5 --from_scratch > logs/nohup_custom_yolov10l_OWOD_t1_scratch_lr00005_lf02.log 2>&1 &

## V11
# Best results  runs_OWOD/20250517_1728_owod_t1_yolo11l_from_scratch/weights/best.pt
nohup python custom_training.py --model yolo11 -e 300 --lr 0.0003 --lrf 0.2 --owod_task t1 --model_size l --devices 0 --dataset owod --batch_size 16 -cl_ms 10 --workers 12 --val_every 5 --from_scratch > logs/nohup_custom_yolo11l_OWOD_t1_scratch_lr00003_lf02.log 2>&1 &
# Options
# nohup python custom_training.py --model yolo11 -e 300 --lr 0.0003 --lrf 0.2 --owod_task t1 --model_size l --devices 0 --dataset owod --batch_size 16 -cl_ms 10 --workers 12 --val_every 5 --from_scratch > logs/nohup_custom_yolo11l_OWOD_t1_scratch_lr00003_lf02.log 2>&1 &
# nohup python custom_training.py --model yolo11 -e 300 --lr 0.0005 --lrf 0.2 --owod_task t1 --model_size l --devices 0 --dataset owod --batch_size 16 -cl_ms 10 --workers 12 --val_every 5 --from_scratch > logs/nohup_custom_yolo11l_OWOD_t1_scratch_lr00005_lf02.log 2>&1 &
# nohup python custom_training.py --model yolo11 -e 300 --lr 0.001 --lrf 0.2 --owod_task t1 --model_size l --devices 0 --dataset owod --batch_size 16 -cl_ms 10 --workers 12 --val_every 5 --from_scratch > logs/nohup_custom_yolo11l_OWOD_t1_scratch_lr0001_lf02.log 2>&1 &
nohup python custom_training.py --model yolo11 -e 300 --lr 0.001 --lrf 0.15 --owod_task t1 --model_size l --devices 2 --dataset owod --batch_size 16 -cl_ms 10 --workers 12 --val_every 5 --from_scratch > logs/nohup_custom_yolov11l_OWOD_t1_scratch_lr0001_lf015.log 2>&1 &
# Medium
nohup python custom_training.py --model yolo11 -e 300 --lr 0.0003 --lrf 0.2 --owod_task t1 --model_size m --devices 0 --dataset owod --batch_size 16 -cl_ms 10 --workers 12 --val_every 5 --from_scratch > logs/nohup_custom_yolo11m_OWOD_t1_scratch_lr00003_lf02.log 2>&1 &

# V12
# Best results  runs_OWOD/20250523_0011_owod_t1_yolo12l_from_scratch/weights/best.pt
nohup python custom_training.py --model yolo12 -e 300 --lr 0.0003 --lrf 0.5 --owod_task t1 --model_size l --devices 0 --dataset owod --batch_size 14 -cl_ms 10 --workers 12 --val_every 5 --from_scratch > logs/nohup_custom_yolo12l_OWOD_t1_scratch_lr00003_lf05.log 2>&1 &
# Options
# nohup python custom_training.py --model yolo12 -e 300 --lr 0.0001 --lrf 0.2 --owod_task t1 --model_size l --devices 0 --dataset owod --batch_size 14 -cl_ms 10 --workers 12 --val_every 5 --from_scratch > logs/nohup_custom_yolo12l_OWOD_t1_scratch_lr00001_lf02.log 2>&1 &
# nohup python custom_training.py --model yolo12 -e 300 --lr 0.0002 --lrf 0.2 --owod_task t1 --model_size l --devices 0 --dataset owod --batch_size 14 -cl_ms 10 --workers 12 --val_every 5 --from_scratch > logs/nohup_custom_yolo12l_OWOD_t1_scratch_lr00002_lf02.log 2>&1 &
# nohup python custom_training.py --model yolo12 -e 300 --lr 0.0003 --lrf 0.5 --owod_task t1 --model_size l --devices 0 --dataset owod --batch_size 14 -cl_ms 10 --workers 12 --val_every 5 --from_scratch > logs/nohup_custom_yolo12l_OWOD_t1_scratch_lr00003_lf05.log 2>&1 &
# nohup python custom_training.py --model yolo12 -e 300 --lr 0.0003 --lrf 0.2 --owod_task t1 --model_size l --devices 0 --dataset owod --batch_size 14 -cl_ms 10 --workers 12 --val_every 5 --from_scratch > logs/nohup_custom_yolo12l_OWOD_t1_scratch_lr00003_lf02.log 2>&1 &
# nohup python custom_training.py --model yolo12 -e 300 --lr 0.0005 --lrf 0.2 --owod_task t1 --model_size l --devices 0 --dataset owod --batch_size 14 -cl_ms 10 --workers 12 --val_every 5 --from_scratch > logs/nohup_custom_yolo12l_OWOD_t1_scratch_lr00005_lf02.log 2>&1 &
# nohup python custom_training.py --model yolo12 -e 300 --lr 0.001 --lrf 0.2 --owod_task t1 --model_size l --devices 0 --dataset owod --batch_size 14 -cl_ms 10 --workers 12 --val_every 5 --from_scratch > logs/nohup_custom_yolo12l_OWOD_t1_scratch_lr0001_lf02.log 2>&1 &
# Medium
nohup python custom_training.py --model yolo12 -e 300 --lr 0.0003 --lrf 0.5 --owod_task t1 --model_size m --devices 2 --dataset owod --batch_size 16 -cl_ms 10 --workers 12 --val_every 5 --from_scratch > logs/nohup_custom_yolo12m_OWOD_t1_scratch_lr00003_lf05.log 2>&1 &