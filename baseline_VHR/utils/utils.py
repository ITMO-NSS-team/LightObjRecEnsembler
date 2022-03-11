from copy import deepcopy

from baseline_VHR.validation_weights import calculate_coco_metrics
from baseline_VHR.visualization import plot_img_bbox
from baseline_VHR.validation_weights import apply_nms
import torchvision.transforms as transforms


def visualise_model_prediction(prediction, target, img, image_id, save, show, path_prediction, title, add_name:str):
    pred = deepcopy(prediction)
    pred['boxes'] = pred['boxes'].cpu().numpy()
    metrics = calculate_coco_metrics(target, pred)
    plot_img_bbox(img, pred, title=title, save=save,
                  image_id=image_id, show=show, path=path_prediction, add_name=add_name)
    return metrics


def visualise_model_prediction_nms(prediction, target, img, image_id, save, show, path_prediction, title, add_name:str):
    nms_prediction = apply_nms(prediction, iou_thresh=0.05)
    nms_prediction['boxes'] = nms_prediction['boxes'].cpu().numpy()
    nms_prediction['labels'] = nms_prediction['labels'].cpu().numpy()
    nms_prediction['scores'] = nms_prediction['scores'].cpu().numpy()
    plot_img_bbox(img, nms_prediction, title=title, save=save,
                  image_id=image_id, show=show, path=path_prediction, add_name=add_name)
    metrics = calculate_coco_metrics(target, nms_prediction)
    return nms_prediction, metrics

def get_model_metrics(prediction, target):
    pred = deepcopy(prediction)
    pred['boxes'] = pred['boxes'].cpu().numpy()
    metrics = calculate_coco_metrics(target, pred)
    return metrics


def get_model_metrics_nms(prediction, target):
    nms_prediction = apply_nms(prediction, iou_thresh=0.05)
    nms_prediction['boxes'] = nms_prediction['boxes'].cpu().numpy()
    nms_prediction['labels'] = nms_prediction['labels'].cpu().numpy()
    nms_prediction['scores'] = nms_prediction['scores'].cpu().numpy()
    metrics = calculate_coco_metrics(target, nms_prediction)
    return nms_prediction, metrics



def torch_to_pil(img):
        """
        Method converts image to PILimgae

        :param img - input image

        :return PIL image
        """
        return transforms.ToPILImage()(img).convert('RGB')