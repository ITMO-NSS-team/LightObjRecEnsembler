from typing import List

import numpy as np

from baseline_VHR.bounding_box import BoundingBox
from baseline_VHR.evaluators.utils.enumerators import BBType


def filtering_ensemble(predictions: List, weights: List, image_id: str,
                       threshold_iou: float = 0.2, threshold_weights: float = 0.6):
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

    # Convert predictions to BoundingBoxes (need refactor)
    bbs = []
    for prediction in predictions:
        bb = []
        for i in range(len(prediction['labels'].tolist())):
            class_id = prediction['labels'].tolist()[i]
            box = prediction['boxes'].tolist()[i]
            x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
            bounding_box = BoundingBox(image_name=image_id, class_id=class_id, coordinates=(x, y, width, height),
                                       bb_type=BBType.DETECTED, confidence=prediction['scores'].tolist()[i])
            bb.append(bounding_box)
        bbs.append(bb)

    predictions = bbs

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
                    intermediate_bbs.append((bb_2, index_model))
                else:
                    new_prediction.append(bb_2)
            predictions[index_model] = new_prediction

        weights_array = []
        for pair in intermediate_bbs:
            i = pair[1]
            weights_array.append(weights[i])

        weights_sum = 0
        if len(weights_array) == 1:
            weights_sum = weights_array[0]
        else:
            for i, w in enumerate(sorted(weights_array)):
                weights_sum += w * (2 ** i)

        if weights_sum >= threshold_weights:
            for pair in intermediate_bbs:
                bb, bb_index = pair
                results[bb_index].append(bb)

        if not predictions[index_max_weight]:
            index_max_len = np.argmax(list(map(len, predictions)))
            index_max_weight = index_max_len

    # Convert BoundingBoxes to prediction.
    for index, result in enumerate(results):
        boxes = [box.get_absolute_bounding_box(False) for box in result]
        scores = [box.get_confidence() for box in result]
        labels = [box.get_class_id() for box in result]
        results[index] = {'boxes': boxes, 'scores': scores, 'labels': labels}

    return results
