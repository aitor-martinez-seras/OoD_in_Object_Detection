from pathlib import Path

from ultralytics import YOLO


def main():
    # Load a model
    
    # model = YOLO("yolov8n.yaml", task='detect')  # build a new model from scratch
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    # model.train(data="coco128.yaml", epochs=3)  # train the model
    # metrics = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    
    # results = model("https://www.imfdb.org/images/thumb/b/b6/AB-46.jpg/300px-AB-46.jpg") 
    # results = model("https://www.imfdb.org/images/b/bc/AO63.jpg") 
    # results = model("https://www.imfdb.org/images/thumb/7/7c/Kalashnikov2020_TT-33-1.jpg/600px-Kalashnikov2020_TT-33-1.jpg")

    # results = model.predict(source=Path('datasets/pruebas_soldado'), save=True)
    # results = model.predict(source=Path('datasets/pruebas_armas'), save=True)
    # results = model.predict(source=Path('datasets/vehiculos_militares'), save=True)
    results = model.predict(source=Path('datasets/proba'), save=True)
    print(results)
    print(results[0].extra_item)


    ##################################
    ## Results es una lista con 2*n elementos, donde n es el numero de imagenes de nuestro dataset.
    ## Los resultados de cada foto vienen en indices consecutivos: [0,1], [2,3], [4,5], ...
    ## En el primer indice estan los resultados originales de YOLO. En el segundo indice esta lo que nosotros queremos sacar
    ## En el siguiente print hago la prueba de lo que digo arriba

    # for i in range(0, len(results)):
    #     if i % 2 == 0:
    #         print('Result normal')
    #         print(i)
    #         print(results[i])
    #     else:
    #         print('Result añadido')
    #         print(i)
    #         print(results[i])
    
    # path = model.export(format="onnx")  # export the model to ONNX format


if __name__ == "__main__":
    main()