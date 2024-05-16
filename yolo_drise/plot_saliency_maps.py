import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse


def plot_saliency_map(img_name, saliency_map_dir='/saliency_maps'):
    """
    Plots the saliency map for a given image name.

    Parameters:
    - img_name: Name of the image to plot the saliency map for.
    - saliency_map_dir: Directory where the saliency maps are stored.
    """
    saliency_map_path = f'{saliency_map_dir}/{img_name}.npy'
    saliency = np.load(saliency_map_path)
    saliency = saliency[795]
    
    plt.figure(figsize=(5, 5))
    plt.imshow(saliency, cmap='jet', alpha=0.5)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(f'Saliency Map for {img_name}')
    plt.axis('off')
    plt.show()

def plot_saliency_on_image(height, width, img_dir, img_name, saliency_map_dir='saliency_maps', target_class_id=0):
    """
    Plots the saliency map overlaid on the original image.

    Parameters:
    - height, width: Dimensions to which the original image is resized.
    - img_path: Path to the original image.
    - img_name: Name of the image to plot the saliency map for.
    - saliency_map_dir: Directory where the saliency maps are stored.
    - target_class_id: ID of the class we are interested to plot the saliency maps for.
    """

    # Load the saliency map
    saliency_map_path = f'{saliency_map_dir}/saliency_map_{img_name}_{target_class_id}.npy'
    saliency = np.load(saliency_map_path)
    
    # Load the original image
    img_path = f'{img_dir}/{img_name}.png'
    original_img = Image.open(img_path)
    # resize it according to the dimensions in which the heatmaps where created!
    resized_img = original_img.resize((height,width), Image.LANCZOS)
    img_np = np.array(resized_img)

    # *** Plotting
    plt.figure(figsize=(5, 5))
    plt.title(f'Saliency Map for {img_name} and target class id:{target_class_id}')
    # 1. Display the original image
    plt.imshow(img_np)
    plt.axis('off')
    # 2.Overlay the saliency map
    plt.imshow(saliency, cmap='jet', alpha=0.5)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.show()

def plot_saliency_on_images(height, width, img_dir, img_name, saliency_map_dir='saliency_maps', target_class_ids=[0]):
    """
    Plots the saliency maps overlaid on the original image for a list of target class IDs,
    displaying all in a single figure with subplots.

    Parameters:
    - height, width: Dimensions to which the original image is resized.
    - img_dir: Directory where the original images are stored.
    - img_name: Name of the image to plot the saliency maps for.
    - saliency_map_dir: Directory where the saliency maps are stored.
    - target_class_ids: List of target class IDs to plot the saliency maps for.
    """
    # Load and resize the original image
    img_path = f'{img_dir}/{img_name}.png'
    original_img = Image.open(img_path)
    resized_img = original_img.resize((height, width), Image.LANCZOS)
    img_np = np.array(resized_img)
    
    # Determine the number of subplots needed
    n = len(target_class_ids)
    plt.figure(figsize=(15, 5 * n))
    
    for i, target_class_id in enumerate(target_class_ids,1):
        # Load the saliency map for the current class ID
        saliency_map_path = f'{saliency_map_dir}/saliency_map_{img_name}_{target_class_id}.npy'
        saliency = np.load(saliency_map_path)
        
        # Plot the original image and overlay the saliency map
        plt.subplot(n, 2, 2*i-1)
        plt.imshow(img_np)
        plt.axis('off')
        plt.title(f'Original: Class ID {target_class_id}')
        
        plt.subplot(n, 2, 2*i)
        plt.imshow(img_np)
        plt.imshow(saliency, cmap='jet', alpha=0.5)  # Overlay the saliency map
        plt.axis('off')
        plt.title(f'Saliency: Class ID {target_class_id}')
        plt.colorbar(fraction=0.046, pad=0.04, ax=plt.gca())
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot saliency maps in different ways.')
    parser.add_argument('--mode', type=str, default='overlay', choices=['single', 'overlay','subplots'], help='Plot mode: single, overlay, or subplots')
    parser.add_argument('--img_name', default='ski', type=str, help='Image name to plot')
    parser.add_argument('--saliency_map_dir', default='saliency_maps', type=str, help='Directory of saliency maps')
    parser.add_argument('--target_class_ids', type=int, nargs='+', help='List of target class IDs for plotting', default=[795])
    parser.add_argument('--height', type=int, default=224, help='Height for resizing the image')
    parser.add_argument('--width', type=int, default=224, help='Width for resizing the image')
    parser.add_argument('--img_dir', default='data', type=str, help='Directory of the images')
    args = parser.parse_args()

    if args.mode == 'single':
        plot_saliency_map(img_name=args.img_name, saliency_map_dir=args.saliency_map_dir)
        # plot_saliency_map(img_name='ski', saliency_map_dir='saliency_maps')
    elif args.mode == 'overlay':
        plot_saliency_on_image(height=args.height, width=args.width, 
                               img_dir=args.img_dir, img_name=args.img_name, 
                               saliency_map_dir=args.saliency_map_dir, target_class_id=args.target_class_ids[0])
        # plot_saliency_on_image(height=224, width=224, 
        #                        img_dir='data', img_name='ski', 
        #                        saliency_map_dir='saliency_maps', target_class_id=795)
    elif args.mode == 'subplots':
        plot_saliency_on_images(height=args.height, width=args.width, 
                                img_dir=args.img_dir, img_name=args.img_name, 
                                saliency_map_dir=args.saliency_map_dir, target_class_ids=args.target_class_ids)
        # plot_saliency_on_images(height=224, width=224, 
        #                 img_dir='data', img_name='ski', 
        #                 saliency_map_dir='saliency_maps', target_class_ids=[746, 795, 796])
        
    

