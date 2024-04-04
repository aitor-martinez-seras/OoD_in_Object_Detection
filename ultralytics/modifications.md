# Where to modify for extracting activations from the internals of the model

[BasePredictor](yolo/engine/predictor.py) (```ultralytics/yolo/engine/predictor.py```): in ```stream_inference``` is where the dataset is looped, calling the batches and the preprocessing, processing and postprocessing steps. In the processing part, where the model is called, we arrive to the next object.

[BaseModel](nn/tasks.py) (```ultralytics/nn/tasks.py```): called by ```BasePredictor```, contains the functions where the model makes the forward pass throught the layers and where you can take the activations out of the layer you want. Specifically follow the calls ```forward -> predict -> _predict_once```. We have passed throught the wanted activations as a variable called ```output_extra``` that comes out to the next object, the ```DetectionPredictor```.


[DetectionPredictor](yolo/v8/detect/predict.py) (```ultralytics/yolo/v8/detect/predict.py```): inherits from ```BasePredictor``` and modifies the postprocessing step. There is where we can take the ```output_extra``` and handle as we want to include it in the ```Results``` object as we want.

# Where to modify training

[BaseTrainer](yolo/engine/trainer.py): in line 270, ```_do_train``` function has the loop over the batches and the validation logic.

## For adding new arguments to training

You have to modify [default.yaml](yolo/cfg/default.yaml) config, adding the new argument. Then when calling ```.train()``` in ```YOLO``` type models, you can add the argument or you can modify it in your custom ```.yaml``` config file

# To add custom dataset classes properly

1. Create a ```.yaml``` config file where the ```dataset_class``` attribute must be included, defining there some flag for the dataset class
2. Create the dataset class inside the ```ultralytics/yolo/data``` folder, either in a separate file or in ```ultralytics/yolo/data/dataset.py```. Ensure to at least define the ```get_img_files```, ```update_labels_info```, ```get_labels``` and ```build_transforms``` if inheriting from ```BaseDataset```. An example of a custom class can be found in [tao.py](yolo/data/tao.py)
3. Create the class inheriting from [BaseDataset](yolo/data/base.py) or from [YOLODataset](yolo/data/dataset.py)
4. Create a function in ```ultralytics/yolo/data/build.py``` imitating ```build_yolo_dataset``` function. Import the dataset class created following ```YOLODataset``` convention.
6. Update ```ultralytics/yolo/data/__init__.py``` with the new imports (dataset class and function to build it)
7. Add in [DetectionTrainer](yolo/v8/detect/train.py) an if statement to catch the flag indicated in the first step and call the build function created. Example:

    ```python
    # Choose dataset type based on YAML file data
    if self.data.get('dataset_class') == 'TAODataset':
        return build_tao_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)
    elif self.data.get('dataset_class') == 'FilteredYOLODataset':
        return build_filtered_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)
    else:
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)
    ```

    ## OWOD 

    For the addition of the OWOD benchmark:
    - To not force the copying of all the COCO and Pascal VOC datasets to the OWOD folder, we modified the method ```get_img_files``` from [BaseDataset](yolo/data/build.py) to catch the parent path from the ```path``` variable inside the ```.yaml``` files of the datasets configuration instead of inferring it by taking the parent of the ```.txt``` file where the paths are being taking from (```train.txt``` for example).
    - To enable the usage of the different tasks, we add a parameter to the default configuration ([default.yaml](yolo/cfg/default.yaml)), called `owod_task`, that can be then set in the configuration passed to the dataset and dataloader creator or the train method. Then this parameter is used in the dataset class to load the correct paths and labels for the task.