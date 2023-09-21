from pathlib import Path

from ultralytics import YOLO


def main():
    # Load a model
    
    # model = YOLO("yolov8n.yaml", task='detect')  # build a new model from scratch
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    # model.train(data="coco128.yaml", epochs=3)  # train the model
    # metrics = model.val()  # evaluate model performance on the validation set

    # results = model.predict(source=Path('datasets/proba'), save=True)
    results = model.predict(source=Path('datasets/coco128'), save=True)
    # print(results)
    # print(results[0].extra_item) ## Para sacar el extra_item que metemos dentro de Results.


    # path = model.export(format="onnx")  # export the model to ONNX format


if __name__ == "__main__":
    main()