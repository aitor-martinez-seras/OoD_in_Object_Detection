import torch
from .rise import RISE
from tqdm import tqdm

class RISEObjectDetection(RISE):
    def __init__(self, model, input_size, device='cpu', p1=0.1, gpu_batch=100):
        super(RISEObjectDetection, self).__init__(model, input_size, device, p1, gpu_batch)


    def process_in_batches(self, stack):
        """
        Process the masked images in batches and handle variable detection outputs.
        Collects results in a list to accommodate varying numbers of detections.
        """
        results = []  # Initialize an empty list to hold the detection results
        
        with torch.no_grad():  # Ensure no gradients are computed
            for i in tqdm(range(0, self.N, self.gpu_batch)):
                # print('processed {}/{}'.format(i, self.N))
                
                # Process the batch through the model
                batch_results = self.model(stack[i:min(i + self.gpu_batch, self.N)])  

                # Instead of concatenating, append each batch's results to the list
                results.extend(batch_results)
        
        return results

    def calculate_saliency_map(self, p, target_class_index, H, W, aggregation='max'):
        """
        Calculate the saliency map based on object detection scores for a target class.
        Overrides the method from RISE to adapt it for object detection.
        """
        N = self.N
        saliency_map = torch.zeros((H, W), device=self.device)
        
        for i in range(N): 
            # Assuming each detection includes 'boxes', 'scores', and 'labels'
            scores = p[i]['scores']
            labels = p[i]['labels']
            
            # Filter scores by target class
            target_scores = [score.item() for score, label in zip(scores, labels) if label == target_class_index]

            # Skip if no detections for target class
            if not target_scores:
                continue

            # Aggregate scores based on specified method
            if aggregation == 'max':
                score_aggregated = max(target_scores)
            elif aggregation == 'avg':
                score_aggregated = sum(target_scores) / len(target_scores)
            else:
                raise ValueError("Unsupported aggregation method. Choose 'max' or 'average'.")

            # Apply aggregated score to mask contribution
            mask_contribution = self.masks[i, 0, :, :] * score_aggregated
            saliency_map += mask_contribution

        # Normalize by N * self.p1 to maintain consistent scaling across configurations
        # saliency_map /= (N * self.p1)
        saliency_map /= saliency_map.max()

        return saliency_map.cpu().numpy()

    def forward(self, x, target_class_indices):
        """
        Forward pass adapted for object detection to generate saliency maps for multiple classes.
        
        Parameters:
        - x: Input image tensor.
        - target_class_indices: A list of class indices for which to generate saliency maps.
        - aggregation: Method for aggregating scores across detections ('max' or 'average').
        
        Returns:
        A dictionary of saliency maps, keyed by the target class indices.
        """
        _, _, H, W = x.size()
        # Apply masks to the image
        stack = self.apply_masks(x)
        # Process in batches
        p = self.process_in_batches(stack)
        
        # Initialize a dictionary to store saliency maps for each target class
        saliency_maps = {}
        
        # Calculate a saliency map for each class in target_class_indices
        for target_class_index in target_class_indices:
            sal = self.calculate_saliency_map(p, target_class_index, H, W)
            saliency_maps[target_class_index] = sal
        
        return saliency_maps