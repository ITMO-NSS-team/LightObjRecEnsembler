from copy import deepcopy
import gc
import itertools
import os

import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms

import baseline_VHR.torch_utils.transforms as T
from baseline_VHR.data_loaders import train_test_split, VHRDataset
from baseline_VHR.visualization import plot_img_bbox
from baseline_VHR.faster_RCNN_baseline import get_fasterRCNN_resnet
from baseline_VHR.utils.ensemble import Rectangle
from baseline_VHR.filtering_ensembel import filtering_ensemble
from baseline_VHR.validation_weights import validation_weights, apply_nms, calculate_coco_metrics


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def torch_to_pil(img):
    return transforms.ToPILImage()(img).convert('RGB')


def rectangle_intersect(first_box: list,
                        second_box: list):
    a = Rectangle(*first_box)
    b = Rectangle(*second_box)
    area = a & b
    if area is None:
        return area, None
    else:
        intersect_area = area.calculate_area()
        # composite_bbox = a - area
        composite_bbox = a.difference(area)
        ratio_1, ratio_2 = intersect_area / a.calculate_area(), intersect_area / b.calculate_area()
        return (ratio_1, ratio_2), composite_bbox


def ensemble_OD_predictions(bboxes: list,
                            labels: list,
                            scores: list,
                            image: np.ndarray,
                            weights: list,
                            area_threshold: float = 0.75):
    best_model_ind = np.argmax(weights)
    weak_model_ind = 1 if best_model_ind == 0 else 0

    bboxes_merged = []
    labels_merged = []
    scores_merged = []

    for index_1, first_box in enumerate(bboxes[weak_model_ind]):
        for index_2, second_box in enumerate(bboxes[best_model_ind]):
            ratio, intersect_coord = rectangle_intersect(first_box, second_box)
            if ratio is None:
                check_flag = True
            elif sum(ratio) > area_threshold:
                CW1 = val_weights[best_model_ind] * scores[weak_model_ind][index_1]
                CW2 = val_weights[weak_model_ind] * scores[best_model_ind][index_2]
                if labels[weak_model_ind][index_1] == labels[best_model_ind][index_2]:
                    chosen_box = second_box
                    chosen_label = labels[best_model_ind][index_2]
                    chosen_score = scores[best_model_ind][index_2]
                else:
                    best_CW_ind = np.argmax([CW1, CW2])
                    chosen_box = first_box if best_CW_ind == 0 else second_box
                    chosen_ind = index_1 if best_CW_ind == 0 else index_2
                    chosen_label = labels[best_CW_ind][chosen_ind]
                    chosen_score = scores[best_CW_ind][chosen_ind]

                check_flag = False
                break

        if check_flag:
            bboxes_merged.append(chosen_box)
            labels_merged.append(chosen_label)
            scores_merged.append(chosen_score)

    compose_bbox = list(itertools.chain(bboxes[best_model_ind], bboxes_merged)),
    compose_labels = list(itertools.chain(labels[best_model_ind], labels_merged))
    compose_scores = list(itertools.chain(scores[best_model_ind], scores_merged))

    return {'boxes': np.array(compose_bbox[0]), 'labels': np.array(compose_labels), 'scores': np.array(compose_scores)}


num_classes = 11
params = {'BATCH_SIZE': 32,
          'LR': 0.001,
          'PRECISION': 32,
          'CLASSES': num_classes,
          'SEED': 42,
          'PROJECT': 'Heads',
          'EXPERIMENT': 'heads',
          'MAXEPOCHS': 500,
          'BACKBONE': 'fasterrcnn_resnet50_fpn',
          'FPN': False,
          'ANCHOR_SIZE': ((32, 64, 128, 256, 512),),
          'ASPECT_RATIOS': ((0.5, 1.0, 2.0),),
          'MIN_SIZE': 1024,
          'MAX_SIZE': 1024,
          'IMG_MEAN': [0.485, 0.456, 0.406],
          'IMG_STD': [0.229, 0.224, 0.225],
          'IOU_THRESHOLD': 0.5
          }

model = get_fasterRCNN_resnet(num_classes=params['CLASSES'],
                              backbone_name=params['BACKBONE'],
                              anchor_size=params['ANCHOR_SIZE'],
                              aspect_ratios=params['ASPECT_RATIOS'],
                              fpn=params['FPN'],
                              min_size=params['MIN_SIZE'],
                              max_size=params['MAX_SIZE'])

params = {'BATCH_SIZE': 32,
          'LR': 0.001,
          'PRECISION': 32,
          'CLASSES': num_classes,
          'SEED': 42,
          'PROJECT': 'Heads',
          'EXPERIMENT': 'heads',
          'MAXEPOCHS': 500,
          'BACKBONE': 'resnet18',
          'ANCHOR_SIZE': ((32, 64, 128, 256, 512),),
          'ASPECT_RATIOS': ((0.5, 1.0, 2.0),),
          'MIN_SIZE': 1024,
          'MAX_SIZE': 1024,
          'IMG_MEAN': [0.485, 0.456, 0.406],
          'IMG_STD': [0.229, 0.224, 0.225],
          'IOU_THRESHOLD': 0.5
          }

model_2 = get_fasterRCNN_resnet(num_classes=params['CLASSES'],
                                backbone_name=params['BACKBONE'],
                                anchor_size=params['ANCHOR_SIZE'],
                                aspect_ratios=params['ASPECT_RATIOS'],
                                min_size=params['MIN_SIZE'],
                                max_size=params['MAX_SIZE'])

save = True
show = False
dataset, dataset_test, dataset_val = train_test_split(VHRDataset, validation_flag=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
filepath = "../local/VHR_statedict_fasterrcnn_resnet50_fpn.pth"
filepath_2 = "../local/VHR_statedict_resnet18.pth"
path = os.path.dirname(os.path.abspath(__file__))
path_prediction = os.path.join(path, 'NWPU VHR-10 dataset', 'new_predictions')

if __name__ == '__main__':
    model.load_state_dict(torch.load(filepath))
    model_2.load_state_dict(torch.load(filepath_2))

    model.eval()
    model_2.eval()

    val_weights = validation_weights([model, model_2], dataset_val)
    # val_weights = [1, 0]

    columns = [
        "AP",
        "AP50",
        "AP75",
        "APsmall",
        "APmedium",
        "APlarge",
        "AR1",
        "AR10",
        "AR100",
        "ARsmall",
        "ARmedium",
        "ARlarge"
    ]
    image_ids = []
    results_resnet50 = pd.DataFrame(columns=columns)
    results_resnet18 = pd.DataFrame(columns=columns)
    results_resnet50_nms = pd.DataFrame(columns=columns)
    results_resnet18_nms = pd.DataFrame(columns=columns)
    results_ensemble_nms = pd.DataFrame(columns=columns)

    for i in range(len(dataset_test)):
        result_current_image = pd.DataFrame()
        img, target = dataset_test[i]
        image_id = str(target['image_id'].tolist()[0] + 1)
        image_ids.append(image_id)
        plot_img_bbox(img, target, title='IMAGE', save=save,
                      image_id=image_id, show=show, path=path_prediction)

        with torch.no_grad():
            prediction = model([img.to(device)])[0]
            prediction_2 = model_2([img.to(device)])[0]

        ### VISUALISE MODELS PREDICTIONS
        pred = deepcopy(prediction)
        pred['boxes'] = pred['boxes'].cpu().numpy()
        metrics_50 = calculate_coco_metrics(target, pred)
        plot_img_bbox(img, pred, title='RESNET50', save=save,
                      image_id=image_id, show=show, path=path_prediction)

        pred = deepcopy(prediction_2)
        pred['boxes'] = pred['boxes'].cpu().numpy()
        metrics_18 = calculate_coco_metrics(target, pred)
        plot_img_bbox(img, pred, title='RESNET18', save=save,
                      image_id=image_id, show=show, path=path_prediction)

        ### VISUALISE MODELS PREDICTIONS AFTER IOU_THERESHOLD
        nms_prediction = apply_nms(prediction, iou_thresh=0.01)
        nms_prediction_2 = apply_nms(prediction_2, iou_thresh=0.01)

        nms_prediction['boxes'] = nms_prediction['boxes'].cpu().numpy()
        nms_prediction_2['boxes'] = nms_prediction_2['boxes'].cpu().numpy()
        nms_prediction['labels'] = nms_prediction['labels'].cpu().numpy()
        nms_prediction_2['labels'] = nms_prediction_2['labels'].cpu().numpy()
        nms_prediction['scores'] = nms_prediction['scores'].cpu().numpy()
        nms_prediction_2['scores'] = nms_prediction_2['scores'].cpu().numpy()

        plot_img_bbox(img, nms_prediction, title='RESNET50_NMS', save=save,
                      image_id=image_id, show=show, path=path_prediction)
        metrics_50_nms = calculate_coco_metrics(target, nms_prediction)

        plot_img_bbox(img, nms_prediction_2, title='RESNET18_NMS', save=save,
                      image_id=image_id, show=show, path=path_prediction)
        metrics_18_nms = calculate_coco_metrics(target, nms_prediction_2)

        ### VISUALISE MODELS PREDICTIONS AFTER ENSEMBLING
        nms_prediction, nms_prediction_2 = filtering_ensemble([nms_prediction, nms_prediction_2], val_weights, image_id)

        compose_bbox = ensemble_OD_predictions([nms_prediction['boxes'], nms_prediction_2['boxes']],
                                               [nms_prediction['labels'], nms_prediction_2['labels']],
                                               [nms_prediction['scores'], nms_prediction_2['scores']],
                                               image=img,
                                               weights=val_weights)
        plot_img_bbox(img, compose_bbox, title='ENSEMBLE', save=save,
                      image_id=image_id, show=show, path=path_prediction)
        metrics_ensemble_nms = calculate_coco_metrics(target, compose_bbox)
        # print("ENSEMBLE:")
        # print(metrics_ensemble_nms)

        results_resnet50 = results_resnet50.append(metrics_50, ignore_index=True)
        results_resnet18 = results_resnet18.append(metrics_18, ignore_index=True)
        results_resnet18_nms = results_resnet18_nms.append(metrics_18_nms, ignore_index=True)
        results_resnet50_nms = results_resnet50_nms.append(metrics_50_nms, ignore_index=True)
        results_ensemble_nms = results_ensemble_nms.append(metrics_ensemble_nms, ignore_index=True)

        ### SAVE METRICS FOR CURRENT IMAGE
        result_current_image['resnet50'] = list(metrics_50.values())
        result_current_image['resnet18'] = list(metrics_18.values())
        result_current_image['resnet18_nms'] = list(metrics_18_nms.values())
        result_current_image['resnet50_nms'] = list(metrics_50_nms.values())
        result_current_image['ensemble'] = list(metrics_ensemble_nms.values())
        result_current_image.index = columns
        result_current_image.to_csv(os.path.join(path_prediction, image_id, f'{image_id}.csv'))

        # TODO apply_nms after ensemble

    results_resnet50['image_id'] = image_ids
    results_resnet18['image_id'] = image_ids
    results_resnet18_nms['image_id'] = image_ids
    results_resnet50_nms['image_id'] = image_ids
    results_ensemble_nms['image_id'] = image_ids

    results_resnet50.to_csv(os.path.join(path_prediction, 'results_resnet50.csv'))
    results_resnet18_nms.to_csv(os.path.join(path_prediction, 'results_resnet18_nms.csv'))
    results_resnet18.to_csv(os.path.join(path_prediction, 'results_resnet18.csv'))
    results_resnet50_nms.to_csv(os.path.join(path_prediction, 'results_resnet50_nms.csv'))
    results_ensemble_nms.to_csv(os.path.join(path_prediction, 'results_ensemble_nms.csv'))
    gc.collect()
