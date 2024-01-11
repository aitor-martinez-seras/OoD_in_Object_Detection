from pathlib import Path
from datetime import datetime


from ultralytics import YOLO
from ultralytics.yolo.utils.callbacks import tensorboard
from ultralytics.yolo.utils import yaml_load
from ultralytics.yolo.utils import DEFAULT_CFG_PATH

def main():
    # Constants
    NOW = datetime.now().strftime("%Y%m%d_%H%M")

    # Load a model
    model = YOLO("yolov8n.yaml", task='detect')  # build a new model from scratch
    # model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Define configuration file using the number of classes
    NC = 30
    yaml_file = f"OAK_{NC}_classes.yaml"
    oak_dataset_config_file = yaml_load(DEFAULT_CFG_PATH.parent / yaml_file, append_filename=True)

    # Add tensorboard callbacks
    tensorboard.writer = tensorboard.SummaryWriter
    for k, v in tensorboard.callbacks.items():
        model.add_callback(k, v)

    # Use the model
    # model.train(
    #     # data="OAK_full.yaml",
    #     data="OAK_40_classes.yaml",
    #     # cfg="ultralytics/yolo/cfg/config_for_oak.yaml.yaml",
    #     # cfg="config_for_oak.yaml",
    #     cfg="config_for_oak_adam.yaml",
    #     epochs=150,
    # )

    model.train(
        # data="OAK_full.yaml",
        data=yaml_file,
        # cfg="ultralytics/yolo/cfg/config_for_oak.yaml.yaml",
        # cfg="config_for_oak.yaml",
        cfg="config_for_oak_adam.yaml",
        epochs=150,
        mixup=0.0,
        close_mosaic=20,
        workers=10,
        name=f'{NC}_classes_{NOW}',
    )

    # metrics = model.val()  # evaluate model performance on the validation set

    # results = model.predict(source=Path('datasets/coco5'), save=True)

    # print('*********')
    # print(len(results))
    # print(results[0].extra_item) ## Para sacar el extra_item que metemos dentro de Results.

    # path = model.export(format="onnx")  # export the model to ONNX format


if __name__ == "__main__":
    main()