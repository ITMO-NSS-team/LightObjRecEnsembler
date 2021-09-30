from typing import List

import pandas as pd
import numpy as np
import torch

from baseline_VHR.VHR_experiment import apply_nms, calculate_coco_metrics


def scaling_weights(weights: List) -> List:
    """Scaling weight to scale (0, 1)."""
    res_weights = []
    metrics_weight = [1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]

    for w in weights:
        w = [y for x in [[w[0]], w[3:]] for y in x]

        w = [a * b for a, b in zip(w, metrics_weight)]
        res_weights.append(np.mean(w))

    return list((res_weights - min(res_weights)) / (max(res_weights) - min(res_weights)))


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

            nms_prediction = apply_nms(prediction, iou_thresh=0.01)
            metric_nms = calculate_coco_metrics(target, nms_prediction)

            results = results.append(metric_nms, ignore_index=True)

        weights_unbalance = list(results.apply(np.nanmean).values)
        weights.append(weights_unbalance)

    weights = scaling_weights(weights)
    return weights
