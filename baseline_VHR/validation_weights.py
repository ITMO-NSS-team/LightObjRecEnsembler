from typing import List
from copy import deepcopy

import torch
import torchvision
import numpy as np
import pandas as pd

from bounding_box import BoundingBox
from baseline_VHR.evaluators.utils.enumerators import BBType
from baseline_VHR.evaluators.coco_evaluator import get_coco_summary


def scaling_weights(weights: List) -> List:
    """Scaling weight to scale (0, 1)."""
    res_weights = []
    # Weights for metrics. AP - the first one primary challenge metric.
    metrics_weight = [1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]

    for w in weights:
        w = [a * b for a, b in zip(w, metrics_weight)]
        res_weights.append(np.mean(w))

    return list(res_weights / sum(res_weights))


def validation_weights(models: List, dataset_validation) -> List:
    """ Get models and calculate metrics on validation data. After creating
    weights for each models from 0 -> 1, that sum of all weight give 1."""
    weights = []
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    columns = ["AP", "AP50", "AP75", "APsmall", "APmedium", "APlarge", "AR1",
               "AR10", "AR100", "ARsmall", "ARmedium", "ARlarge"]

    for model in models:
        results = pd.DataFrame(columns=columns)
        model.eval()

        for i in range(len(dataset_validation)):
            img, target = dataset_validation[i]

            with torch.no_grad():
                prediction = model([img.to(device)])[0]

            nms_prediction = apply_nms(prediction)
            metric_nms = calculate_coco_metrics(target, nms_prediction)

            results = results.append(metric_nms, ignore_index=True)

        weights_unbalance = list(results.apply(np.nanmean).values)
        weights.append(weights_unbalance)

    weights = scaling_weights(weights)
    return weights


def apply_nms(orig_prediction, iou_thresh=0.3):
    # torchvision returns the indices of the boxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = deepcopy(orig_prediction)
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction


def calculate_coco_metrics(target, prediction):
    gt_bbs = []
    detected_bbs = []

    image_name = str(target['image_id'].tolist()[0] + 1)

    for i in range(len(target['labels'].tolist())):
        class_id = target['labels'].tolist()[i]
        box = target['boxes'].tolist()[i]
        x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
        bounding_box = BoundingBox(image_name=image_name, class_id=class_id,
                                   coordinates=(x, y, width, height), bb_type=BBType.GROUND_TRUTH)
        gt_bbs.append(bounding_box)

    for i in range(len(prediction['labels'].tolist())):
        class_id = prediction['labels'].tolist()[i]
        box = prediction['boxes'].tolist()[i]
        x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
        bounding_box = BoundingBox(image_name=image_name, class_id=class_id, coordinates=(x, y, width, height),
                                   bb_type=BBType.DETECTED, confidence=prediction['scores'].tolist()[i])
        detected_bbs.append(bounding_box)

    coco_summary = get_coco_summary(gt_bbs, detected_bbs)
    return coco_summary
