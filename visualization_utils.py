from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional

import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
from ultralytics.yolo.engine.results import Results


def create_folder(folder_path: Path, now: str, ood_method_name: str = '') -> Path:
    """Create and return the path to store images."""
    directory_name = f'{now}_{ood_method_name}' if ood_method_name else now
    folder_path = folder_path / directory_name
    folder_path.mkdir(exist_ok=True)
    return folder_path


# Save the original image 
def save_image_from_results_and_data(results: List[Results], data: Dict[str, Tensor], c: int):
    for i, res in enumerate(results):
        imgs_path = Path("figures/images")
        imgs_path.mkdir(exist_ok=True)
        #img_to_plot = res.orig_img[i].permute(1, 2, 0).cpu().numpy()
        # Extract the image to plot using the ori_shape from data
        img_to_plot = res.orig_img[i].permute(1, 2, 0).cpu().numpy()
        padded_height = (640 - data['ori_shape'][i][0]) // 2
        padded_width = (640 - data['ori_shape'][i][1]) // 2
        img_to_plot = img_to_plot[padded_height:padded_height+data['ori_shape'][i][0], padded_width:padded_width+data['ori_shape'][i][1]]
        plt.imshow(img_to_plot)
        plt.axis('off')
        plt.savefig(imgs_path / f'img_{c + i:03d}.png', bbox_inches='tight', pad_inches=0)
        plt.close()


def prepare_bboxes_and_labels(
        results: Results, class_names: List[str], ood_decision: Optional[List[int]], targets: Optional[Dict[str, Tensor]],
        img_idx: int, valid_preds_only: bool, possible_unk_boxes: Optional[List[Tensor]] = None, ood_decision_on_possible_unk_boxes: Optional[List[List[int]]] = None,
        distances_unk_prop_per_image: Optional[List[np.ndarray]] = None):
    """Prepare bounding boxes and labels for plotting."""
    if valid_preds_only and hasattr(results, 'valid_preds'):
        # Filter bounding boxes and labels based on valid predictions
        valid_preds = results.valid_preds.cpu()
        bboxes = results.boxes.xyxy.cpu()[valid_preds]
        labels = results.boxes.cls.cpu()[valid_preds]
        confs = results.boxes.conf.cpu()[valid_preds]
    else:
        bboxes = results.boxes.xyxy.cpu()
        labels = results.boxes.cls.cpu()
        confs = results.boxes.conf.cpu()

    labels_str = [f'{class_names[int(lbl.item())]} - {conf:.2f}' for lbl, conf in zip(labels, confs)]
    colors = ['green'] * len(labels)  # Default color for in-distribution

    if ood_decision:
        # Modify colors based on OoD decisions
        for i, decision in enumerate(ood_decision[img_idx]):
            if decision == 0:
                colors[i] = 'red'  # OoD
                colors[i] = 'orange'  # OoD
            elif decision == 1:
                colors[i] = 'green'  # In-Distribution

    if targets:
        # Append target bounding boxes and labels
        bboxes = torch.cat((bboxes, targets["bboxes"][img_idx]), dim=0)
        target_labels = targets["cls"][img_idx]
        # If the target is cls 80, then use the color pink else purple
        labels_str.extend([f'{class_names[int(lbl.item())]} - GT' for lbl in target_labels])
        colors.extend(['pink' if lbl == 80 else 'purple' for lbl in target_labels])

    if possible_unk_boxes:
        # Append possible unknown bounding boxes and labels
        # Extract the possible unknown boxes and the OoD decision from the list
        one_img_possible_unk_boxes = possible_unk_boxes[img_idx] 
        one_img_ood_decision_on_possible_unk_boxes = ood_decision_on_possible_unk_boxes[img_idx] if ood_decision_on_possible_unk_boxes else None
        # Append the possible unknown boxes to the tensor of boxes
        bboxes = torch.cat((bboxes, one_img_possible_unk_boxes), dim=0)
        # If the OoD decision is provided, use the decision for the string and color the possible unknown boxes
        if one_img_ood_decision_on_possible_unk_boxes:
            if distances_unk_prop_per_image:
                # Append the distance to the string
                #labels_str.extend([f'PROP - UNK - {dist:.2f}' if one_img_ood_decision_on_possible_unk_boxes[_box_idx] == 0 else f'PROP - IN-DIST - {dist:.2f}' for _box_idx, dist in zip(range(len(one_img_possible_unk_boxes)), distances_unk_prop_per_image[img_idx])])
                labels_str.extend([f'{dist:.2f}' if one_img_ood_decision_on_possible_unk_boxes[_box_idx] == 0 else f'PROP - IN-DIST - {dist:.2f}' for _box_idx, dist in zip(range(len(one_img_possible_unk_boxes)), distances_unk_prop_per_image[img_idx])])
            else:
                labels_str.extend([f'UNK' if one_img_ood_decision_on_possible_unk_boxes[_box_idx] == 0 else f'PROP - IN-DIST' for _box_idx in range(len(one_img_possible_unk_boxes))])
            colors.extend(['yellow' if one_img_ood_decision_on_possible_unk_boxes[_box_idx] == 0 else 'orange' for _box_idx in range(len(one_img_possible_unk_boxes))])
        else:
            # Append the distance to the string if available
            if distances_unk_prop_per_image:
                #labels_str.extend([f'PROP - {dist:.2f}' for dist in distances_unk_prop_per_image[img_idx]])
                labels_str.extend([f'{dist:.2f}' for dist in distances_unk_prop_per_image[img_idx]])
            else:
                labels_str.extend([f'PROP'] * len(one_img_possible_unk_boxes))
            colors.extend(['yellow'] * len(one_img_possible_unk_boxes))

        # for i, boxes in enumerate(one_img_possible_unk_boxes):
        #     bboxes = torch.cat((bboxes, boxes), dim=0)
        #     # If the OoD decision is provided, use the decision for the possible unknown boxes
        #     if ood_decision_on_possible_unk_boxes:
        #         labels_str.extend([f'UNK PROP - UNK' if one_img_ood_decision_on_possible_unk_boxes[i] == 0 else f'UNK PROP - IN-DIST' for _ in range(len(boxes))])
        #         colors.extend(['yellow' if one_img_ood_decision_on_possible_unk_boxes[i] == 0 else 'orange'] * len(boxes))
        #     else:
        #         labels_str.extend([f'UNK PROP'] * len(boxes))
        #         colors.extend(['yellow'] * len(boxes))    

    return bboxes, labels_str, colors


def plot_bounding_boxes(img: Tensor, bboxes: Tensor, labels: List[str], colors: List[str], width: int, font: str, font_size: int, image_path: Path, use_labels: bool,
                        plot_gray_bands: bool, original_shape: List[Tuple[int, int]] = None):
    """Plot and save the image with bounding boxes."""
    if not use_labels:
        labels = None
    im = draw_bounding_boxes(
        img.cpu(),
        bboxes,
        width=width,
        font=font,
        font_size=font_size,
        labels=labels,
        colors=colors
    )
    # Remove padding if plot_gray_bands is False
    if not plot_gray_bands:
        # Limit the axis to the original shape using the actual shape as reference
        max_shape = max(im.shape[1:])
        width_padding = (max_shape - original_shape[1]) // 2
        height_padding = (max_shape - original_shape[0]) // 2
        if (max_shape - original_shape[1]) % 2 != 0:
            width_padding -= 1
        if (max_shape - original_shape[0]) % 2 != 0:
            height_padding -= 1
        print(f"Original shape: {original_shape}")
        print(f"Current shape: {im.shape}")
        print(f"Padding: {width_padding}, {height_padding}")
        im = im[:, height_padding:original_shape[0]+height_padding, width_padding:original_shape[1]+width_padding]
    plt.imshow(im.permute(1, 2, 0))
    # plt.xlim(width_padding, original_shape[0])
    # plt.ylim(original_shape[1], height_padding)
    plt.axis('off')
    plt.savefig(image_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_results(
        class_names: List[str],
        results: List[Results],
        folder_path: Path,
        now: str,
        valid_preds_only: bool,
        origin_of_idx: int,
        image_format: str = 'pdf',
        ood_decision: Optional[List[int]] = None,
        ood_method_name: str = '',
        targets: Optional[Dict[str, Tensor]] = None,
        possible_unk_boxes: Optional[List[Tensor]] = None,
        ood_decision_on_possible_unk_boxes: Optional[List[List[int]]] = None,
        distances_unk_prop_per_image: Optional[List[np.ndarray]] = None,
        use_labels: bool = True,
        plot_gray_bands=False,
        original_shapes=None,
    ):
    """Main function to plot results."""
    width = 3
    font = 'FreeMonoBold'
    font_size = 12

    if not plot_gray_bands and original_shapes is None:
        raise ValueError("The original_shapes must be provided when plot_gray_bands is False.")

    assert not (ood_decision and not ood_method_name), "If ood_decision is provided, ood_method_name must not be empty."
    
    output_folder = create_folder(folder_path, now, ood_method_name)

    for img_idx, res in enumerate(results):

        bboxes, labels_str, colors = prepare_bboxes_and_labels(
            res,
            class_names,
            ood_decision,
            targets,
            img_idx,
            valid_preds_only,
            possible_unk_boxes,
            ood_decision_on_possible_unk_boxes,
            distances_unk_prop_per_image
        )

        image_path = output_folder / f'{(origin_of_idx + img_idx):03}.{image_format}'
        plot_bounding_boxes(res.orig_img[img_idx], bboxes, labels_str, colors, width, font, font_size, image_path, use_labels, plot_gray_bands, original_shapes[img_idx])
