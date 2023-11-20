from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import draw_bounding_boxes

def plot_results(class_names, results, folder_path: Path, now: str, valid_preds_only: bool, origin_of_idx: int, image_format='pdf', ood_decision=None, ood_method_name=''):
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
