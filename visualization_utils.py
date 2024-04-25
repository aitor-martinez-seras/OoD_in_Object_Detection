from pathlib import Path
from typing import List, Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torchvision.utils import draw_bounding_boxes

from ultralytics.yolo.engine.results import Results


def plot_results(
        class_names: List[str],
        results: List[Results],
        folder_path: Path,
        now: str,
        valid_preds_only: bool,
        origin_of_idx: int,
        image_format: str = 'pdf',
        ood_decision: Union[List[int], None] = None,
        ood_method_name: str = '',
        targets: Union[Dict[str, Tensor], None] = None,
        possible_unk_boxes_and_decisions: Union[Tuple[List[np.ndarray], List[np.ndarray]], None] = None
    ):
    # ----------------------
    ### Codigo para dibujar las cajas predichas y los targets ###
    # ----------------------
    # Parametros para plot
    width = 2
    font = 'FreeMonoBold'
    font_size = 12

    if ood_decision:
        assert ood_method_name is not None, "If ood_decision exists, ood_method must be a string with the name of the OoD method"

    # Create folder to store images
    if ood_method_name:
        prueba_ahora_path = folder_path / f'{now}_{ood_method_name}'
    else:
        prueba_ahora_path = folder_path / now
    prueba_ahora_path.mkdir(exist_ok=True)

    for img_idx, res in enumerate(results):
        
        if valid_preds_only:
            valid_preds = np.array(res.valid_preds)
            bboxes = res.boxes.xyxy.cpu()[valid_preds]
            labels = res.boxes.cls.cpu()[valid_preds]
        else:
            bboxes = res.boxes.xyxy.cpu()
            labels = res.boxes.cls.cpu()

            # Este codigo es por si queremos que las cajas OoD se pinten con nombre OoD
            # for i, decision in enumerate(ood_decision[img_idx]):
            #     if decision == 0:
            #         labels[i] = 0
            # class_names[0] = 'OoD'

            # labels_for_plot = []
            # for i, lbl in enumerate(labels):
            #     if lbl == 0:
            #         labels_for_plot.append(f'{class_names[int(n.item())]} - {res.boxes.conf[i]:.2f}')
            #     else:    
            #         labels_for_plot.append(f'{class_names[int(n.item())]} - {res.boxes.conf[i]:.2f}') 
        
        # If we have OoD labels, we plot the boxes with green for In-Distribution and red for OoD
        if ood_decision:
            if targets:
                # Append bboxes of the targets and its labels
                # Expand the ood_decision to the labels and assign the number n = -5 to the targets
                # Then paint the targets in violet
                
                bboxes = torch.cat((bboxes, targets["bboxes"][img_idx]), dim=0)
                labels = torch.cat((labels, targets["cls"][img_idx]), dim=0)
                ood_decision_one_image = ood_decision[img_idx]
                ood_decision_one_image = np.append(ood_decision_one_image, [-5]*len(targets["cls"][img_idx]))
                colors = []
                for decision in ood_decision_one_image:
                    if decision == 1:
                        colors.append('green')
                    elif decision == -5:
                        colors.append('violet')
                    elif decision == 0:
                        colors.append('red')
                    else:
                        raise ValueError("The OoD decision must be 0, 1 or -5")
                    
                labels_str = []
                for i, lbl in enumerate(labels):
                    if ood_decision_one_image[i] == -5:
                        labels_str.append(f'{class_names[int(lbl.item())]} - GT')
                    else:
                        labels_str.append(f'{class_names[int(lbl.item())]} - {res.boxes.conf[i]:.2f}')

                im = draw_bounding_boxes(
                    res.orig_img[img_idx].cpu(),
                    bboxes,
                    width=width,
                    font=font,
                    font_size=font_size,
                    labels=labels_str,
                    colors=colors
                )
            else:
                im = draw_bounding_boxes(
                    res.orig_img[img_idx].cpu(),
                    bboxes,
                    width=width,
                    font=font,
                    font_size=font_size,
                    labels=[f'{class_names[int(n.item())]} - {res.boxes.conf[i]:.2f}' for i, n in enumerate(labels)],
                    colors=['red' if n == 0 else 'green' for n in ood_decision[img_idx]]
                )
        
        # Just plot normal predictions
        else:
            if targets:
                bboxes_preds_plus_gt = torch.cat((bboxes, targets["bboxes"][img_idx]), dim=0)
                #labels_preds_plus_gt = torch.cat((labels, targets["cls"][img_idx]), dim=0)
                # import random
                # get_colors = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]
                # colors = get_colors(len(labels))
                #colors.extend(['violet']*len(targets["cls"][img_idx]))
                colors = ['green']*len(labels) + ['violet']*len(targets["cls"][img_idx])
                
                labels_str = []
                for i, lbl in enumerate(labels):
                    # Check if they are predictions  for this image
                    labels_str.append(f'{class_names[int(lbl.item())]} - {res.boxes.conf[i]:.2f}')
                for i, lbl in enumerate(targets["cls"][img_idx]):
                    labels_str.append(f'{class_names[int(lbl.item())]} - GT')
                im = draw_bounding_boxes(
                    res.orig_img[img_idx].cpu(),
                    bboxes_preds_plus_gt,
                    width=width,
                    font=font,
                    font_size=font_size,
                    labels=labels_str,
                    colors=colors
                )
            else:
                im = draw_bounding_boxes(
                    res.orig_img[img_idx].cpu(),
                    bboxes,
                    width=5,
                    font=font,
                    font_size=font_size,
                    labels=[f'{class_names[int(n.item())]} - {res.boxes.conf[i]:.2f}' for i, n in enumerate(labels)]
                )

        plt.imshow(im.permute(1,2,0))
        plt.savefig(prueba_ahora_path / f'{(origin_of_idx + img_idx):03}.{image_format}', dpi=300)
        plt.close()



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


def prepare_bboxes_and_labels(
        results: Results, class_names: List[str], ood_decision: Optional[List[int]], targets: Optional[Dict[str, Tensor]],
        img_idx: int, valid_preds_only: bool, possible_unk_boxes: Optional[List[Tensor]] = None, ood_decision_on_possible_unk_boxes: Optional[List[int]] = None):
    """Prepare bounding boxes and labels for plotting."""
    if valid_preds_only and hasattr(results, 'valid_preds'):
        # Filter bounding boxes and labels based on valid predictions
        valid_preds = results.valid_preds[img_idx].cpu()
        bboxes = results.boxes.xyxy[img_idx].cpu()[valid_preds]
        labels = results.boxes.cls[img_idx].cpu()[valid_preds]
        confs = results.boxes.conf[img_idx].cpu()[valid_preds]
    else:
        bboxes = results.boxes.xyxy[img_idx].cpu()
        labels = results.boxes.cls[img_idx].cpu()
        confs = results.boxes.conf[img_idx].cpu()

    labels_str = [f'{class_names[int(lbl.item())]} - {conf:.2f}' for lbl, conf in zip(labels, confs)]
    colors = ['green'] * len(labels)  # Default color for in-distribution

    if ood_decision:
        # Modify colors based on OoD decisions
        for i, decision in enumerate(ood_decision[img_idx]):
            if decision == 0:
                colors[i] = 'red'  # OoD
            elif decision == 1:
                colors[i] = 'green'  # In-Distribution

    if targets:
        # Append target bounding boxes and labels
        bboxes = torch.cat((bboxes, targets["bboxes"][img_idx]), dim=0)
        target_labels = targets["cls"][img_idx]
        labels_str.extend([f'{class_names[int(lbl.item())]} - GT' for lbl in target_labels])
        colors.extend(['violet'] * len(target_labels))  # Color for ground truth

    if possible_unk_boxes:
        # Append possible unknown bounding boxes and labels
        for i, boxes in enumerate(possible_unk_boxes):
            bboxes = torch.cat((bboxes, boxes), dim=0)
            labels_str.extend([f'UNK PROP'] * len(boxes))
            colors.extend(['yellow'] * len(boxes))    

    return bboxes, labels_str, colors



def plot_bounding_boxes(img: Tensor, bboxes: Tensor, labels: List[str], colors: List[str], width: int, font: str, font_size: int, image_path: Path):
    """Plot and save the image with bounding boxes."""
    im = draw_bounding_boxes(
        img.cpu(),
        bboxes,
        width=width,
        font=font,
        font_size=font_size,
        labels=labels,
        colors=colors
    )
    plt.imshow(im.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
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
        ood_decision_on_possible_unk_boxes: Optional[List[int]] = None, 
    ):
    """Main function to plot results."""
    width = 2
    font = 'FreeMonoBold'
    font_size = 12

    assert not (ood_decision and not ood_method_name), "If ood_decision is provided, ood_method_name must not be empty."
    
    output_folder = create_folder(folder_path, now, ood_method_name)

    for img_idx, res in enumerate(results):
        bboxes, labels_str, colors = prepare_bboxes_and_labels(
            res, class_names, ood_decision, targets, img_idx, valid_preds_only, possible_unk_boxes, ood_decision_on_possible_unk_boxes
        )

        image_path = output_folder / f'{(origin_of_idx + img_idx):03}.{image_format}'
        plot_bounding_boxes(res.orig_img[img_idx], bboxes, labels_str, colors, width, font, font_size, image_path)
