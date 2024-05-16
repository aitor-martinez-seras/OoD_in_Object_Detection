# Black box (perturbation-based) XAI for Object Detection

## RISE
For RISE, we can use any classifier. In this case, the example is based on ResNet50
```
python rise_resnet50.py --device cuda:0 --img_name ski --top_k 3
```
**Parameters**:
- `img_name`: Specifies the input image.
- `top_k`: Indicates the top K predictions to highlight.

**Output Example**
<div align="center">
  <img src="data/ski_rise.png" width="400" height="690">
</div>

## O-RISE
It is an adaptation of RISE for object detectors. Considering a target class as reference, we take all the bounding boxes provided by the Object Detector, and we only consider those (scores) that are identified with the same class as the one set as reference. This is, given a target class, we take the scores when the Object Detector has identified a bounding box like that class, and we use that score to weight that mask. 

_Note that for a single image, we can get multiple bounding boxes, and we can have more than one identifying the bounding box as the same class. Therefore, we need to select how to deal in such situations in order to weight properly the mask at hand._

In this implementation, we have considered the option of either averaging or just taking the maximum score for the target class. 

An example to run this method would be as follows:
```
python orise_faster_rcnn.py --device cuda:2 --img_name ski --target_classes 1,35 --gpu_batch 8
```
**Parameters**:
- `target_classes`: Classes for which saliency maps are generated. Indices 1 and 35 correspond to "person" and "skis", respectively.
- `gpu_batch`: Adjust to prevent GPU memory overflow.

**Output Example**
<div align="center">
  <img src="data/ski_orise.png" width="500" height="500">
</div>

Another example for a more complex image (gandalf) would be as follows:
```
python orise_faster_rcnn.py --device cuda:2 --img_name gandalf --target_classes 1,72,73,74,76 --gpu_batch 8
```
being the target classes
- 72: tv
- 73: latptop
- 74: mouse
- 76: keyboard

<div align="center">
  <img src="data/gandalf_orise.png" width="400" height="900">
</div>

## D-RISE
Actual implementation for Object Detectors. It considers both the classification and localization information (the latter through the target ground truth bounding box).

Unlike other approaches, it requires to specify the target bounding box and the target class.
_Currently the code only supports one class and bounding box per run._

It can be executed as:
```
python drise_faster_rcnn.py --device cuda:2 --img_name gandalf --target_xml gandalf --target_classes 1,74 --gpu_batch 16
```
being now necessary a `target_xml_file` (located at annotations folder) where we encode the target bounding box.

The heatmaps are clearly different if we compare against O-RISE. For instance, for the ski image (for both "person"(1) and "skis"(35) targets/objects):
<div align="center">
  <img src="data/ski_drise_person.png" width="400" height="200">
</div>
<div align="center">
  <img src="data/ski_drise_skis.png" width="400" height="200">
</div>

Similarly, for gandalf image for "person"(1) and "mouse"(74) targets:
<div align="center">
  <img src="data/gandalf_drise_person.png" width="400" height="200">
</div>
<div align="center">
  <img src="data/gandalf_drise_mouse.png" width="400" height="200">
</div>


