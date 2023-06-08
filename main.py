from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.yolo.utils import yaml_load

def main():
    # Load a model
    
    # model = YOLO("yolov8n.yaml", task='detect')  # build a new model from scratch
    model = YOLO("yolov8n.pt", task='detect')
    
    model.val('VisDrone.yaml')

    # To save the state dict only from the Sequential object of Pytorch
    # state_dict = YOLO("yolov8n.pt").model.model.state_dict()  # load a pretrained model (recommended for training)
    # torch.save(state_dict, 'state_dict_yolov8n.pt')

    # OPTION 1 of loading.
    # model = YOLO("yolov8n.yaml", task='detect')
    # model.load('yolov8n.pt')

    # OPTION 2
    # model = YOLO("yolov8n.yaml", task='detect')
    # model.model.model.load_state_dict(state_dict)

    # Either way, class names must be updated
    # coco_dataset_info = yaml_load('coco128_custom.yaml')
    # model.names.update(coco_dataset_info['names'])

    # Use the model
    # model.train(data="coco128.yaml", epochs=3)  # train the model
    # metrics = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

    # Diferent PREDICTS
    # results = model("https://www.imfdb.org/images/thumb/b/b6/AB-46.jpg/300px-AB-46.jpg") 
    # results = model("https://www.imfdb.org/images/b/bc/AO63.jpg") 
    # results = model("https://www.imfdb.org/images/thumb/7/7c/Kalashnikov2020_TT-33-1.jpg/600px-Kalashnikov2020_TT-33-1.jpg")
    results = model.predict(source=Path('datasets/pruebas'), save=True)
    # results = model.predict(source='https://www.youtube.com/watch?v=mx7AVi3RNeA', save=True)
    
    # path = model.export(format="onnx")  # export the model to ONNX format

 
if __name__ == "__main__":
    main()
