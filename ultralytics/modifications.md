# Where to modify for extracting activations from the internals of the model

[BasePredictor](yolo/engine/predictor.py) (```ultralytics/yolo/engine/predictor.py```): in ```stream_inference``` is where the dataset is looped, calling the batches and the preprocessing, processing and postprocessing steps. In the processing part, where the model is called, we arrive to the next object.

[BaseModel](nn/tasks.py) (```ultralytics/nn/tasks.py```): called by ```BasePredictor```, contains the functions where the model makes the forward pass throught the layers and where you can take the activations out of the layer you want. Specifically follow the calls ```forward -> predict -> _predict_once```. We have passed throught the wanted activations as a variable called ```output_extra``` that comes out to the next object, the ```DetectionPredictor```.


[DetectionPredictor](yolo/v8/detect/predict.py) (```ultralytics/yolo/v8/detect/predict.py```): inherits from ```BasePredictor``` and modifies the postprocessing step. There is where we can take the ```output_extra``` and handle as we want to include it in the ```Results``` object as we want.
