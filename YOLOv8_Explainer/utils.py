from pathlib import Path

import cv2
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt


def letterbox(
    im: np.ndarray,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    """
    Resize and pad image while meeting stride-multiple constraints.

    Args:
        im (numpy.ndarray): Input image.
        new_shape (tuple, optional): Desired output shape. Defaults to (640, 640).
        color (tuple, optional): Color of the border. Defaults to (114, 114, 114).
        auto (bool, optional): Whether to automatically determine padding. Defaults to True.
        scaleFill (bool, optional): Whether to stretch the image to fill the new shape. Defaults to False.
        scaleup (bool, optional): Whether to scale the image up if necessary. Defaults to True.
        stride (int, optional): Stride of the sliding window. Defaults to 32.

    Returns:
        numpy.ndarray: Letterboxed image.
        tuple: Ratio of the resized image.
        tuple: Padding sizes.

    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    #r = (new_shape[0] / shape[0], new_shape[1] / shape[1])  Hecho por mi
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    #new_unpad = int(round(shape[1] * r[1])), int(round(shape[0] * r[0]))  hecho por mi
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    
    return im, ratio, (dw, dh)

def display_images(images):
    """
    Display a list of PIL images in a grid.

    Args:
        images (list[PIL.Image]): A list of PIL images to display.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, len(images), figsize=(15, 7))
    if len(images) == 1:
        axes = [axes]
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis('off')
    plt.show()

def save_images(imgs, folder="explainability_images", name="image.png"):
    """Takes as input a numpy value and stores an image"""
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)
    img_names = folder_path / name
    for i, img in enumerate(imgs):
        if isinstance(img, Image.Image):
            detections_image = img
        elif isinstance(img, np.ndarray):
            detections_image = Image.fromarray(img)
        elif img is None:
            continue
        else:
            raise ValueError("Input should be a numpy array or a PIL Image.")
        # save image
        if detections_image.mode != 'RGB':
            detections_image = detections_image.convert('RGB')
        detections_image.save((folder_path / (img_names.stem + f'_{i}.jpg')).as_posix())