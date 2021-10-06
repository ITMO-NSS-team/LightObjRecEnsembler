from copy import deepcopy

from baseline_VHR.validation_weights import calculate_coco_metrics
from baseline_VHR.visualization import plot_img_bbox
from baseline_VHR.validation_weights import apply_nms


def visualise_model_prediction(prediction, target, img, image_id, save, show, path_prediction, title):
    pred = deepcopy(prediction)
    pred['boxes'] = pred['boxes'].cpu().numpy()
    metrics = calculate_coco_metrics(target, pred)
    plot_img_bbox(img, pred, title=title, save=save,
                  image_id=image_id, show=show, path=path_prediction)
    return metrics


def visualise_model_prediction_nms(prediction, target, img, image_id, save, show, path_prediction, title):
    nms_prediction = apply_nms(prediction, iou_thresh=0.1)
    nms_prediction['boxes'] = nms_prediction['boxes'].cpu().numpy()
    nms_prediction['labels'] = nms_prediction['labels'].cpu().numpy()
    nms_prediction['scores'] = nms_prediction['scores'].cpu().numpy()
    plot_img_bbox(img, nms_prediction, title=title, save=save,
                  image_id=image_id, show=show, path=path_prediction)
    metrics = calculate_coco_metrics(target, nms_prediction)
    return nms_prediction, metrics
