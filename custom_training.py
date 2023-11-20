from pathlib import Path

from ultralytics import YOLO
from ultralytics.yolo.utils.callbacks import tensorboard


def main():
    # Load a model
    
    model = YOLO("yolov8n.yaml", task='detect')  # build a new model from scratch
    # model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    tensorboard.writer = tensorboard.SummaryWriter

    # Add tensorboard callbacks
    for k, v in tensorboard.callbacks.items():
        model.add_callback(k, v)

    # Use the model
    model.train(
        data="OAK_full.yaml",
        #cfg="ultralytics/yolo/cfg/config_for_oak.yaml.yaml",
        cfg="config_for_oak.yaml",
        epochs=75,
    )

    # metrics = model.val()  # evaluate model performance on the validation set

    # results = model.predict(source=Path('datasets/coco5'), save=True)

    # print('*********')
    # print(len(results))
    # print(results[0].extra_item) ## Para sacar el extra_item que metemos dentro de Results.

    # path = model.export(format="onnx")  # export the model to ONNX format


if __name__ == "__main__":
    main()