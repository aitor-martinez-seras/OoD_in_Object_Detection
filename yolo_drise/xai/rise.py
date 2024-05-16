import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from tqdm import tqdm
# from PIL import Image
# import cv2

class RISE(nn.Module):
    def __init__(self, model, input_size, device='cpu', p1=0.1, gpu_batch=100):
        super(RISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.device = device
        self.gpu_batch = gpu_batch
        self.p1 = p1

    def generate_masks(self, N, s, p1, savepath='masks.npy', preload_masks_to_device=False):
        """
        Generate random masks for the RISE algorithm.
        
        Parameters:
        - N: The number of masks to generate.
        - s: The size of the grid that the image is divided into (s x s), stride.
        - p1: The probability of setting a cell in the grid to 1.
        - savepath: The path where the generated masks are saved.
        """
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        # Generate random grid
        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size))
        
        # Generate masks with random shifts
        for i in tqdm(range(N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping (with skimage)
            _upsampling = resize(grid[i], up_size, order=1, mode='reflect',anti_aliasing=False)
            self.masks[i, :, :] = _upsampling[x:x + self.input_size[0], y:y + self.input_size[1]]
            # Linear upsampling and cropping (with PIL)
            # _upsampling = Image.fromarray(grid[i]).resize(up_size.astype(int), Image.BILINEAR)
            # self.masks[i, :, :] = np.array(_upsampling)[x:x + self.input_size[0], y:y + self.input_size[1]]
            # Linear upsampling and cropping (with cv2)
            # _upsampling = cv2.resize(grid[i], (int(up_size[1]), int(up_size[0])), interpolation=cv2.INTER_LINEAR)
            # self.masks[i, :, :] = _upsampling[x:x + self.input_size[0], y:y + self.input_size[1]]
    
        # Reshape and save the masks    
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        np.save(savepath, self.masks)
        # Load masks to the specified device
        self.masks = torch.from_numpy(self.masks).float()
        if preload_masks_to_device:
            self.masks = self.masks.to(self.device)
        self.N = N
        self.p1 = p1

    def load_masks(self, filepath, preload_masks_to_device=False):
        """
        Load masks from a specified file path.
        
        Parameters:
        - filepath: The path from where to load the masks.
        """
        # Load the masks, transfer them to the specified device
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float()
        if preload_masks_to_device:
            self.masks = self.masks.to(self.device)
        self.N = self.masks.shape[0] # Update the number of masks

    def apply_masks(self, x):
        """
        Apply the generated masks to the input image.
        """
        if self.masks.device != x.device:
            stack = torch.mul(self.masks, x.data.to(self.masks.device))
        else:
            stack = torch.mul(self.masks, x.data)
        return stack
 
    def calculate_saliency_map(self, p, H, W):
        """
        Calculate the saliency map-s from model predictions.
        """
        N = self.N
        CL = p.size(1)  # Number of classes
        
        # generate a saliency map per class
        sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W))
        # reshape saliency maps
        sal = sal.view((CL, H, W))
        # normalize
        # sal = sal / N / self.p1 # equivalent to --> (sal) / (N*self.p1)
        sal /= sal.max()
        
        return sal.cpu().numpy()

    def process_in_batches(self, stack):
        """
        Process the masked images in batches to avoid GPU memory overload.
         -In the case of Resnet50 trained on ILSVRC2012, 1000 classes have to be predicted. 
         Therefore, the output (p) is expected to have size of [N, 1000]
        """
        p = []
        with torch.no_grad(): # very important to not build a graph --> VRAM of nvidia is HUGELY consumed!
            for i in tqdm(range(0, self.N, self.gpu_batch)):
                print('processed {}/{}'.format(i, self.N))
                batch_output = self.model(stack[i:min(i + self.gpu_batch, self.N)])
                p.append(batch_output)
     
        p = torch.cat(p,dim=0)
        return p

    def forward(self, x):
        """
        The forward pass applies masks, processes inputs in batches, and calculates the saliency map.
        """
        _, _, H, W = x.size()
        # Apply masks to the image
        stack = self.apply_masks(x)
        # Process in batches
        p = self.process_in_batches(stack)
        # Calculate saliency map
        saliency_maps = self.calculate_saliency_map(p, H, W)
        
        return saliency_maps
    
    
class RISEBatch(RISE):
    def forward(self, x):
        # Apply array of filters to the image
        N = self.N
        B, C, H, W = x.size()
        stack = torch.mul(self.masks.view(N, 1, H, W), x.data.view(B * C, H, W))
        stack = stack.view(B * N, C, H, W)
        stack = stack

        #p = nn.Softmax(dim=1)(model(stack)) in batches
        p = []
        for i in range(0, N*B, self.gpu_batch):
            p.append(self.model(stack[i:min(i + self.gpu_batch, N*B)]))
        p = torch.cat(p)
        CL = p.size(1)
        p = p.view(N, B, CL)
        sal = torch.matmul(p.permute(1, 2, 0), self.masks.view(N, H * W))
        sal = sal.view(B, CL, H, W)
        return sal