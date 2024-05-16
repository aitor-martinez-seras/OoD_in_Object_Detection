from typing import List, Optional

import torch
import numpy as np
from tqdm import tqdm

from .rise import RISE


class DRISE(RISE):
    def __init__(self, model, input_size, device ='cpu', p1=0.1, gpu_batch=100):
        super(DRISE, self).__init__(model, input_size, device, p1, gpu_batch)
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch
        self.p1 = p1

    def yolo_api_input(self, x):
        if self.scaled_to_0_1:
            return x * 255  # YOLOv8 divides input by 255 
        else:
            return x

    def yolo_api_output(self, model_output):
        """
        Extracts bounding boxes, scores, and labels from YOLOv5 output.
        """
        results = []
        for one_img_output in model_output:
            boxes = one_img_output.boxes.xyxy.cpu()
            scores = one_img_output.boxes.conf.cpu()
            labels = one_img_output.boxes.cls.cpu().to(torch.int64)
            results.append({'boxes': boxes, 'scores': scores, 'labels': labels})
        return results

    def process_in_batches(self, stack):
        """
        Process the masked images in batches and handle variable detection outputs.
        Collects results in a list to accommodate varying numbers of detections.
        """
        results = []  # Initialize an empty list to hold the detection results
        
        with torch.no_grad():  # Ensure no gradients are computed
            #for i in tqdm(range(0, self.N, self.gpu_batch)):
            total_batches = self.N // self.gpu_batch
            for i in range(0, self.N, self.gpu_batch):
                # Print progress every 25% aproximately (could happend that from 75% to 100% there are less than 25 images to process)
                if i % (total_batches // 4) == 0:
                    print(f'Processed {i}/{total_batches} batches.')
                # Extract a batch of images from the stack and move to the model's device
                batch_input = stack[i:min(i + self.gpu_batch, self.N)]
                if batch_input.device != self.model.device:
                    batch_input = batch_input.to(self.model.device)
                
                # Preprocess the batch input for the model
                batch_input = self.yolo_api_input(batch_input)
                
                # Process the batch through the model
                batch_results = self.model(batch_input, verbose=False)  # Verbose only works with yolov8
                #batch_results = self.model(stack[i:min(i + self.gpu_batch, self.N)])  

                # Extract bounding boxes, scores, and labels from the output
                batch_results = self.yolo_api_output(batch_results)
                
                # Plot one image from the stack
                # import matplotlib.pyplot as plt
                # plt.imshow(batch_input[0].cpu().permute(1, 2, 0))
                # plt.savefig('d_rise_stack.png')
                # plt.close()

                # Instead of concatenating, append each batch's results to the list
                results.extend(batch_results)
        
        return results

    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) of two boxes with format [xmin, ymin, xmax, ymax].
        """
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2

        # Calculate intersection, i.e., area of overlap between the two boxes (could be 0 if boxes don't intersect)
        inter_x1 = max(xmin1, xmin2)
        inter_y1 = max(ymin1, ymin2)
        inter_x2 = min(xmax1, xmax2)
        inter_y2 = min(ymax1, ymax2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        # Calculate the union area by using Formula: Union(A,B) = A + B - Inter(A,B)
        box1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
        box2_area = (xmax2 - xmin2) * (ymax2 - ymin2)
        union_area = box1_area + box2_area - inter_area

        # Compute the IoU
        iou = inter_area / union_area
        return iou


    def calculate_saliency_map(self, p, target_class_index, target_bbox, H, W, aggregation='sum') -> np.ndarray:
        """
        Calculate the saliency map for D-RISE by considering IoU, class scores, and all masks.
        """
        N = self.N  # Number of masks
        saliency_map = torch.zeros((H, W), device=self.device)
        
        contributions = []  # To keep track of all contributions for aggregation
        max_points = 0  # Keep track of the maximum points
        max_contribution = None  # To store the mask contribution with the highest points
        
        for i in range(N):  # Iterate through each processed output corresponding to a mask
            boxes = p[i]['boxes']  # Assuming p[i] has 'boxes', 'scores', and 'labels'
            scores = p[i]['scores']
            labels = p[i]['labels']

            for box, score, label in zip(boxes, scores, labels):
                if label == target_class_index:
                    iou_score = self.calculate_iou(target_bbox, box.cpu().numpy())
                    if iou_score > 0:  # Only consider overlaps
                        points = iou_score * score
                        mask_contribution = self.masks[i, 0, :, :] * points
                        contributions.append(mask_contribution)
                        if points > max_points:
                            max_points = points
                            max_contribution = mask_contribution

        # Move saliency map to cpu
        if saliency_map.device != 'cpu':
            saliency_map = saliency_map.to('cpu')

        # Apply selected aggregation method
        if contributions:
            
            if aggregation == 'sum':
                for contribution in contributions:
                    saliency_map += contribution
            elif aggregation == 'avg':
                for contribution in contributions:
                    saliency_map += contribution
                saliency_map /= len(contributions)
            elif aggregation == 'max' and max_contribution is not None:
                saliency_map += max_contribution

        # ***Normalization
        # a. consistent scaling
        # saliency_map /= (N * self.p1)
        
        # b. by max for visualization
        saliency_map /= saliency_map.max()
        
        return saliency_map.cpu().numpy()

    def forward(self, x, target_class_indices=None, target_bbox=None, results: Optional[List] = None, mode: str ='visualization'):
            """
            Forward pass adapted for object detection to generate saliency maps for multiple classes.
            
            Parameters:
            - x: Input image tensor.
            - target_class_indices: A list of class indices for which to generate saliency maps.
            - aggregation: Method for aggregating scores across detections ('max' or 'average').
            
            Returns:
            A dictionary of saliency maps, keyed by the target class indices.
            """
            # For YOLO API
            self.scaled_to_0_1 = False if x.max() > 1 else True

            if mode == 'visualization':
                _, _, H, W = x.size()
                # Apply masks to the image
                stack = self.apply_masks(x)
                # Process in batches
                p = self.process_in_batches(stack)
                
                # Initialize a dictionary to store saliency maps for each target class
                saliency_maps = {}
                
                # Calculate a saliency map for each class in target_class_indices
                for target_class_index in target_class_indices: # code only supports one single class index and bounding box
                    sal = self.calculate_saliency_map(p, target_class_index, target_bbox, H, W)
                    saliency_maps[target_class_index] = sal
                
                return saliency_maps

            elif mode == 'object_detection':
                # We are going to receive a batch of images and we need to return the saliency maps for each image
                # calculating the heatmap for all predicted instances in the image
                M, _, H, W = x.size()
                saliency_maps_per_image = torch.zeros((M, H, W), device='cpu')

                for _i in range(M):
                    # Apply masks to the images
                    stack = self.apply_masks(x[_i].unsqueeze(0))
                    # Process in batches
                    p = self.process_in_batches(stack)
                    # Extract the predictions for the original image
                    image_results = results[_i]
                    preds_cls = image_results.boxes.cls.cpu()
                    preds_boxes = image_results.boxes.xyxy.cpu()
                    # For every bounding box, compute the saliency map for the predicted class of that box
                    saliency_maps_one_img = torch.zeros((H, W), device='cpu')
                    for i in range(len(preds_cls)):
                        target_class_index = preds_cls[i].item()
                        target_bbox = preds_boxes[i].tolist()
                        sal = self.calculate_saliency_map(p, target_class_index, target_bbox, H, W)
                        saliency_maps_one_img += torch.tensor(sal, device='cpu')

                    # Scale to range [0, 1]
                    saliency_maps_one_img = (saliency_maps_one_img - saliency_maps_one_img.min()) / (saliency_maps_one_img.max() - saliency_maps_one_img.min())

                    # Plot the saliency map over the image
                    import matplotlib.pyplot as plt
                    plt.imshow(x[_i].cpu().permute(1, 2, 0))
                    plt.imshow(saliency_maps_one_img.unsqueeze(-1), alpha=0.5, cmap='jet')
                    plt.savefig(f'./yolo_drise/saliency_maps_one_img_{_i}.png')
                    plt.close()
                                        
                    saliency_maps_per_image[_i] = saliency_maps_one_img  #.clone() ???????

                return saliency_maps_per_image
            
            else:
                raise ValueError('mode should be either "visualization" or "object_detection"')