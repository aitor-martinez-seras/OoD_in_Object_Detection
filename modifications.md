# Introduction

This file contains the data about the modifcations that must be made to the ultralytics library in order to make the method work. This modifications where originally made to an early version of the ultralytics library. This version is in ```ultralytics_old```, and here we reference the changes to that library. In the newer versions, the paths and names of the scripts modified are similar. but not equal.

# Where to modify for extracting activations from the internals of the model

## Important

New version of ultralytics enables the usage of an argument called ```embed``` that could be used for this purpose. Its description is as follows:
```python
embed: # (list[int], optional) return feature vectors/embeddings from given layers
```
It is not used in this library as we are porting the old code to the newer versions aiming to do the least modification of the original code. Probably using this new embed functionallity will result in easier implementation.


## Instructions as in the original version of ultralytics

[BasePredictor](ultralytics_old/yolo/engine/predictor.py) (```ultralytics/yolo/engine/predictor.py```): in ```stream_inference``` is where the dataset is looped, calling the batches and the preprocessing, processing and postprocessing steps. In the processing part, where the model is called (the forward pass), we arrive to the next object, the [BaseModel](nn/tasks.py).

configure_extra_output_of_the_model

[BaseModel](ultralytics_old/nn/tasks.py) (```ultralytics/nn/tasks.py```): called by ```BasePredictor```, contains the functions where the model makes the forward pass throught the layers and where you can take the activations out of the layer you want. Specifically follow the calls ```forward -> predict -> _predict_once```. 
Here some logic with an ```if``` statement can be added to take the activations out of the layer you want. This ```if``` statement is controlled in our case by an added attribute to the model called ```.extraction_mode```. The statement has the mode hardcoded, so in case you want to add more modes, you have to modify this ```if``` statement manually in the function.
Then, we pass throught the wanted activations as a variable called ```output_extra``` that comes out to the next object, the ```DetectionPredictor```. 

[DetectionPredictor](ultralytics_old/yolo/v8/detect/predict.py) (```ultralytics/yolo/v8/detect/predict.py```): inherits from ```BasePredictor``` and modifies the postprocessing step. There is where we can take the ```output_extra``` and handle as we want to include it in the ```Results``` object as we want. The results object has been conveniently modified to include the ```output_extra``` in the ```__init__``` method.

In the [ood_utils.py](ultralytics_old/ood_utils.py), in the ```configure_extra_output_of_the_model``` function, it is the logic and the hardcoded strings to control the extraction of the activations.

[```Results```](ultralytics_old/yolo/engine/results.py) (```ultralytics_old/yolo/engine/results.py```): Add the extra_item and the strides attributes to the ```__init__```.


# Config modifications

## Add new args to the default

For OWOD:
- owod_task:  # OWOD task, i.e. t1, t2,... by default not activated
- number_of_classes: 0  # number of classes in the OWOD task. 0 means to use the number of classes in the dataset

For validation:
- val_every: saving time in training with validation only every ```val_every``` steps.


## Add OWOD config

[owod.yaml](ultralytics_old/yolo/cfg/owod.yaml)```ultralytics/yolo/cfg/owod.yaml```

# Training modifications

[BaseTrainer](ultralytics_old/yolo/engine/trainer.py): in line 270, ```_do_train``` function has the loop over the batches and the validation logic.

## Modified for training

### Enable custom datasets

[DetectionTrainer](ultralytics_old/yolo/v8/detect/train.py): in the function ```build_dataset```, enable the loading of the custom dataset class, as in the val.py case.

### Enable dynamically selecting number of classes


[BaseTrainer](ultralytics_old/yolo/engine/trainer.py): inside ```__init__``` in line 145, added code that checks the overrides and changes the number of classes in self.

### val_every

Enables not computing the validation on the validation dataset every epoch, but instead to make it every val_every steps.

- Added check in [BaseTrainer](ultralytics_old/yolo/engine/trainer.py), in lines 368 and 370 for making the val_every argument work. 
- The config in ```ultralytics/yolo/cfg/default.yaml``` must be modified to add the argument val_every.

## For adding new arguments to training

You have to modify [default.yaml](ultralytics_old/yolo/cfg/default.yaml) config, adding the new argument. Then when calling ```.train()``` in ```YOLO``` type models, you can add the argument or you can modify it in your custom ```.yaml``` config file

# Validation modifications

https://github.com/aitor-martinez-seras/yolo-ood/commit/e7e8b0f120e3de714d3b94baa5df78efd5299d67

[val.py](ultralytics_old/yolo/v8/detect/val.py): The custom dataset classes created must be added when building dataset in ```build_dataset``` function. Example:

```python
if self.data.get('dataset_class') == 'TAODataset':
return build_tao_dataset(self.args, img_path, batch, self.data, mode=mode, stride=gs)
```

# Add custom dataset classes properly


## For the first addition

Add the following code to ```ultralytics_old/yolo/data/base.py``` in the update_labels function:
```python
def update_labels(self, include_class: Optional[list]):
    """include_class, filter labels to include only these classes (optional)."""
    include_class_array = np.array(include_class).reshape(1, -1)
    count_removed_boxes = 0
    for i in range(len(self.labels)):
        if include_class is not None:
            cls = self.labels[i]['cls']
            bboxes = self.labels[i]['bboxes']
            segments = self.labels[i]['segments']
            keypoints = self.labels[i]['keypoints']
            j = (cls == include_class_array).any(1)
            count_removed_boxes += len(j) - j.sum()  # To count the number of removed instances
            self.labels[i]['cls'] = cls[j]
            self.labels[i]['bboxes'] = bboxes[j]
            if segments:
                self.labels[i]['segments'] = [segments[si] for si, idx in enumerate(j) if idx]
            if keypoints is not None:
                self.labels[i]['keypoints'] = keypoints[j]
        if self.single_cls:
            self.labels[i]['cls'][:, 0] = 0
    if count_removed_boxes > 0:  # Print the number of removed instances only if there are any
        print(f'Removed {count_removed_boxes} boxes from labels to include only the number of classes defined')
```

## For each addition

1. Create a ```.yaml``` config file that must be included in [cfg](ultralytics_old/yolo/cfg) folder where the ```dataset_class``` attribute must be included, defining there some flag for the dataset class.
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