from pathlib import Path

from ultralytics import YOLO

import os
import torch
import copy
import numpy as np
from torchvision.ops.boxes import box_iou
from ultralytics.yolo.utils.ops import xywhn2xyxy, xywh2xyxy

def main():
    # Load a model
    
    # model = YOLO("yolov8n.yaml", task='detect')  # build a new model from scratch
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    # model.train(data="coco128.yaml", epochs=3)  # train the model
    # metrics = model.val()  # evaluate model performance on the validation set

    # results = model.predict(source=Path('datasets/proba'), save=True)
    results = model.predict(source=Path('datasets/coco5'), save=True)
    # print('*********')

    # print(len(results))
    # print(results[0].extra_item) ## Para sacar el extra_item que metemos dentro de Results.

    # path = model.export(format="onnx")  # export the model to ONNX format



    ##########################################################
    #### AÑADIMOS A LOS RESULTADOS LA CAJA Y LABEL REALES ANOTADOS
    ##########################################################
    # Folder Path
    path = 'C:\\Users\\jonander.rivera\\PycharmProjects\\yolo-pruebas\\datasets\\coco5_label'

    # Change the directory
    os.chdir(path)

    # iterate through all file
    i = 0
    for file in os.listdir():

        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path = f"{path}\{file}"

            # call read text file function
            with open(file_path, 'r') as f:
                leido = f.read()
            leido_newline = leido.split("\n") ## Datos en forma str y [label caja]
            leido_newline = leido_newline[:-1]  ## Quito el ultimo porque es un ''
            leido_lista_numpys = []
            for j in leido_newline:
                leido_lista_numpys.append(np.fromstring(j, dtype=float, sep=' ')) ## Convierto str a un numpy
            leido_numpy = np.array(leido_lista_numpys) ## Creo un array de la lista
            tensor_final = torch.FloatTensor(leido_numpy) ## Creo un tensor del array
            results[i].annotated_label = tensor_final[:, 0]
            alto, ancho = results[i].orig_shape
            results[i].annotated_box = xywhn2xyxy(tensor_final[:, 1:], w=ancho, h=alto)
            results[i].predicted_box = results[i].boxes.xyxy
        i = i + 1


    ##########################################################
    ## CALCULAMOS LOS IOU DE LAS CAJAS PREDICHAS Y RELAES
    ##########################################################

    for n in results:
        annotated = n.annotated_box.to('cuda:0')
        matriz_iou = box_iou(n.predicted_box, annotated)


    print(results[0].annotated_label)
    print(results[0].boxes.cls)
    clases_coinciden = torch.zeros(results[0].boxes.cls.size()[0], results[0].annotated_label.size()[0]) ## Tensor de zeros para rellenar con True False si las clases coinciden
    for i_pred in range(0,results[0].boxes.cls.size()[0]):
        for i_real in range(0,results[0].annotated_label.size()[0]):
            if results[0].boxes.cls[i_pred] == results[0].annotated_label[i_real]:
                clases_coinciden[i_pred,i_real] = True
            else:
                clases_coinciden[i_pred, i_real] = False

    # print('** Clases coinciden **')
    clases_coinciden = clases_coinciden.to('cuda:0')
    # print(clases_coinciden)
    # print('** ANNOTATED LABEL **')
    # print(results[0].annotated_label)
    # print('** ANNOTATED BOX **')
    # print(annotated)
    # print('** PREDICTED LABEL **')
    # print(results[0].boxes.cls)
    # print(results[0].boxes.conf)
    # print('** PREDICTED BOX **')
    # print(results[0].predicted_box)
    # print('** BOX IOU **')
    annotated = results[0].annotated_box.to('cuda:0')
    box_iou_matriz = box_iou(results[0].predicted_box, annotated)
    # print(box_iou_matriz)
    # print('Solo los que coinciden')
    iou_clases_coinciden = (box_iou_matriz * clases_coinciden).cpu()
    iou_clases_coinciden = iou_clases_coinciden.numpy()

    max_iterate = True
    list_finales = []
    copia_iou_clases_coindicen = copy.deepcopy(iou_clases_coinciden)
    while max_iterate:
        ind = np.unravel_index(np.argmax(copia_iou_clases_coindicen, axis=None),
                               copia_iou_clases_coindicen.shape)  # returns tuple where max value is
        max_value = copia_iou_clases_coindicen[ind]
        print(max_value)
        if max_value > 0.5:
            list_finales.append(ind)
            copia_iou_clases_coindicen[:, ind[1]] = 0
            copia_iou_clases_coindicen[ind[0], :] = 0
            print(copia_iou_clases_coindicen)
        else:
            max_iterate = False
    print('** Lista con las boxes que me quedo. (a,b) donde a es el indice de la box predicha y b el indice de la box anotada')
    print(list_finales)
    print(iou_clases_coinciden)





if __name__ == "__main__":
    main()