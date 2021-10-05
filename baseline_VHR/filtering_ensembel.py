from typing import List

import numpy as np

from baseline_VHR.bounding_box import BoundingBox


def filtering_ensemble(predictions: List[List[BoundingBox]], weights: List,
                       threshold_iou: float, threshold_weights: float = None):
    """

    Args:
        predictions: BoundingBoxes.
        weights: ValidationWeights.
        threshold_iou: first hyperparameter, intersection over union between BB.
        threshold_weights: second hyperparameter, sum of weights.

    Returns: filtered list of BoundingBoxes.

    """
    if len(predictions) != len(weights):
        raise IndexError("Quantity of models and weights should be the same.")

    if threshold_weights is None:
        threshold_weights = max(weights)

    results = [[] for _ in range(len(predictions))]
    index_max_weight = np.argmax(weights)

    while predictions[index_max_weight]:
        intermediate_bbs = []
        bb_1 = predictions[index_max_weight].pop()
        intersection = BoundingBox.clone(bb_1)
        intermediate_bbs.append((bb_1, index_max_weight))

        for index_model, prediction in enumerate(predictions):
            new_prediction = []

            for index_bb, bb_2 in enumerate(prediction):
                if (intersection.get_class_id() == bb_2.get_class_id()) and \
                        (BoundingBox.iou(intersection, bb_2) >= threshold_iou):
                    intersection = BoundingBox.get_intersection(intersection, bb_2)
                    intermediate_bbs.append((bb_2, index_bb))
                else:
                    new_prediction.append(bb_2)
            predictions[index_model] = new_prediction

        weights_sum = 0
        for pair in intermediate_bbs:
            i = pair[1]
            if i == index_max_weight:
                weights_sum += weights[i]
            else:
                weights_sum += weights[i] * weights[i]

        if weights_sum >= threshold_weights:
            for pair in intermediate_bbs:
                bb, bb_index = pair
                results[bb_index].append(bb)

        if not predictions[index_max_weight]:
            index_max_len = np.argmax(list(map(len, predictions)))
            index_max_weight = index_max_len

    return results
