# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops


class DetectionPredictor(BasePredictor):

    def postprocess(self, preds, img, orig_imgs):
        """Postprocesses predictions and returns a list of Results objects."""

        ## preds es tupla de 2 elementos:
        ## 1ero, tensor de (1,84,5040). 84 son el box + las 80 clases. 5040 son las boxes.
        ## 2ndo, lista de 3 elementos. En cada elemento tenemos un tensor. Son los feature map segun Aitor
        # output_extra = preds[1]
        output_extra = preds[1][0]
        ## Cojo el [0] porque ahora mismo preds[1] es una lista de 2 elementos donde:
        ## el primer elemento es la salida de la red neuronal
        ## el segundo elemento hay 3 items, con lo que creemos que son los feature map.
        preds = preds[0]

        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes,
                                        extra_item=output_extra)

        output_extra = preds[1]
        # print('++++++++++++ POSTPROCESS ++++++++++++++++')
        # # print(output_extra)
        # print(len(output_extra))
        # for idx, o in enumerate(output_extra):
        #     print(f'Extra item shape: {o.shape}')
        #     print(f'Preds shape: {preds[0][idx].shape}')
        # print('-----------------------------------------------------------------------')

        preds = preds[0]
        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred, extra_item=output_extra[i]))
        return results


def predict(cfg=DEFAULT_CFG, use_python=False):
    """Runs YOLO model inference on input image(s)."""
    model = cfg.model or 'yolov8n.pt'
    source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
        else 'https://ultralytics.com/images/bus.jpg'

    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    predict()
