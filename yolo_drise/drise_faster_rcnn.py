import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

import torchvision.transforms as transforms
import torchvision
import argparse

import xml.etree.ElementTree as ET

from xai.drise import DRISE
from utils.utils import tensor_imshow, get_class_name_coco

# add the following to avoid ssl issues from the server
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


#########################
# Arguments
#########################
parser = argparse.ArgumentParser()
parser.add_argument("--img_name", default='ski')
parser.add_argument("--datadir", default='data/')
parser.add_argument("--device", default='cuda:0')
parser.add_argument("--input_size", default=(224,224))  # Ensure this is handled correctly as a tuple elsewhere
parser.add_argument("--gpu_batch", type=int, default=16)
# masks 
parser.add_argument("--maskdir", default='masks/')
parser.add_argument("--N", type=int, default=6000)
parser.add_argument("--stride", default=8)
parser.add_argument("--p1", default=0.1)

# specific for xml file
parser.add_argument("--annotations_dir", default='annotations/')
parser.add_argument("--target_xml", default='gandalf', help="VOC style annotations 4 elements: [x_min, y_min, x_max, y_max]")
parser.add_argument("--target_classes", default="")


args = parser.parse_args()

# it only accepts a single target class
args.target_classes =  [int(cls) for cls in args.target_classes.split(',')] if args.target_classes != "" else [1]

#########################
# Import data
#########################
img_path = args.datadir + args.img_name
img_np = np.array(Image.open(img_path+'.png' )) # 224,224,3

# preprocessing function
preprocess = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    transforms.Resize((224, 224)),
    # Normalization
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std =[0.229, 0.224, 0.225]),
])

tensor = preprocess(img_np) # 3,224,224
tensor = tensor.unsqueeze(0).to(args.device) # 1,3,224,224


#########################
# Process XML (annotations)
#########################
xml_file_path = args.annotations_dir + args.target_xml + '.xml'
# Load and parse the XML file
tree = ET.parse(xml_file_path)
root = tree.getroot()
# The object you're interested in
target_object = get_class_name_coco(args.target_classes[0])
# Iterate through each object in the XML and extract bndbox parameters for the target object
for obj in root.findall('object'):
    name = obj.find('name').text
    if name == target_object:
        # Extract bndbox parameters
        bndbox = obj.find('bndbox')
        xmin = bndbox.find('xmin').text
        ymin = bndbox.find('ymin').text
        xmax = bndbox.find('xmax').text
        ymax = bndbox.find('ymax').text
        print(f"Found {target_object} with bounding box: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
        
        # define the target bbox
        target_bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
        
        break  # Remove this if you want to find all instances of the target object

#########################
# Resize bounding box
#########################
height_scaling_factor = 224/img_np.shape[0]
width_scaling_factor = 224/img_np.shape[1]

target_bbox[0] *= width_scaling_factor 
target_bbox[1] *= height_scaling_factor
target_bbox[2] *= width_scaling_factor
target_bbox[3] *= height_scaling_factor

#########################
# Load black box model for explanations
#########################
# AQUI LA PARTE DE ULTRALYTICS
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model = model.eval()
model = model.to(args.device)

# To use multiple GPUs
# model = nn.DataParallel(model)


#########################
# Generate Explainer Instance
#########################
explainer = DRISE(model=model, 
                  input_size=args.input_size, 
                  device=args.device,
                  gpu_batch=args.gpu_batch)

# Generate masks for RISE or use the saved ones.
generate_new = True
mask_file = args.maskdir + 'masks_' + args.img_name + '.npy'

if generate_new or not os.path.isfile(mask_file):
    # explainer.generate_masks(N=5000, s=8, p1=0.1, savepath= mask_file)
    explainer.generate_masks(N=args.N, s=args.stride, p1=args.p1, savepath= mask_file)
else:
    explainer.load_masks(mask_file)
    print('Masks are loaded.')

"""
#########################
# Explain(option 1) --> Generate Saliency map(s), save it/them and plot in another script
#########################
# apply xai
saliency = explainer(x=tensor,
                     target_class_indices=args.target_classes,
                     target_bbox=target_bbox)

for k,v in saliency.items():
    saliency_map_path = os.path.join('saliency_maps', f'saliency_map_{args.img_name}_{k}.npy')
    np.save(saliency_map_path, v)
"""

#########################
# Explain & Visualize (option 2) --> no saliency map exports and visualize straight
#########################
# apply xai
saliency = explainer(x=tensor,
                     target_class_indices=args.target_classes,
                     target_bbox=target_bbox)
                     
plt.figure(figsize=(10, 5 * len(args.target_classes)))

for i, cl in enumerate(args.target_classes):
    # Plot original image
    plt.subplot(len(args.target_classes), 2, 2*i + 1)
    plt.axis('off')
    plt.title(get_class_name_coco(cl))
    tensor_imshow(inp=tensor.squeeze(0))
    # plt.imshow(tensor.cpu().numpy().squeeze(0).transpose((1, 2, 0)))

    # Plot saliency map for the class
    plt.subplot(len(args.target_classes), 2, 2*i + 2)
    plt.axis('off')
    plt.title(get_class_name_coco(cl))
    tensor_imshow(inp=tensor.squeeze(0))
    # plt.imshow(tensor.cpu().numpy().squeeze(0).transpose((1, 2, 0)))
    plt.imshow(saliency[cl], cmap='jet', alpha=0.5)
    plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(img_path + '_drise_' + get_class_name_coco(cl) + '.png')
plt.show()
