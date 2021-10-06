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
from baseline_VHR.validation_weights import validation_weights, calculate_coco_metrics
from baseline_VHR.utils.utils import visualise_model_prediction, visualise_model_prediction_nms


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
                            weights: list,
                            area_threshold: float = 0.75):
    best_model_ind = np.argmax(weights)
    weak_model_ind = 1 if best_model_ind == 0 else 0

    bboxes_merged = []
    labels_merged = []
    scores_merged = []

    chosen_box = None
    chosen_label = None
    chosen_score = None
    check_flag = True

    for index_1, first_box in enumerate(bboxes[weak_model_ind]):
        for index_2, second_box in enumerate(bboxes[best_model_ind]):
            ratio, intersect_coord = rectangle_intersect(first_box, second_box)
            if ratio is None:
                check_flag = True
            elif sum(ratio) > area_threshold:
                CW1 = weights[best_model_ind] * scores[weak_model_ind][index_1]
                CW2 = weights[weak_model_ind] * scores[best_model_ind][index_2]
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

        if check_flag and chosen_box is not None:
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

params = {'BATCH_SIZE': 32,
          'LR': 0.001,
          'PRECISION': 32,
          'CLASSES': num_classes,
          'SEED': 42,
          'PROJECT': 'Heads',
          'EXPERIMENT': 'heads',
          'MAXEPOCHS': 500,
          'BACKBONE': 'mobilenet_v3_large',
          'ANCHOR_SIZE': ((32, 64, 128, 256, 512),),
          'ASPECT_RATIOS': ((0.5, 1.0, 2.0),),
          'MIN_SIZE': 1024,
          'MAX_SIZE': 1024,
          'IMG_MEAN': [0.485, 0.456, 0.406],
          'IMG_STD': [0.229, 0.224, 0.225],
          'IOU_THRESHOLD': 0.5
          }

model_3 = get_fasterRCNN_resnet(num_classes=params['CLASSES'],
                              backbone_name=params['BACKBONE'],
                              anchor_size=params['ANCHOR_SIZE'],
                              aspect_ratios=params['ASPECT_RATIOS'],
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
          'BACKBONE': 'densenet121',
          'ANCHOR_SIZE': ((32, 64, 128, 256, 512),),
          'ASPECT_RATIOS': ((0.5, 1.0, 2.0),),
          'MIN_SIZE': 1024,
          'MAX_SIZE': 1024,
          'IMG_MEAN': [0.485, 0.456, 0.406],
          'IMG_STD': [0.229, 0.224, 0.225],
          'IOU_THRESHOLD': 0.5
          }

model_4 = get_fasterRCNN_resnet(num_classes=params['CLASSES'],
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
filepath_3 = "../local/VHR_statedict_mobilenet_v3_large.pth"
filepath_4 = "../local/VHR_statedict_densenet121.pth"
path = os.path.dirname(os.path.abspath(__file__))
path_prediction = os.path.join(path, 'NWPU VHR-10 dataset', 'predictions_4_models')

if __name__ == '__main__':
    model.load_state_dict(torch.load(filepath))
    model_2.load_state_dict(torch.load(filepath_2))
    model_3.load_state_dict(torch.load(filepath_3, map_location=device))
    model_4.load_state_dict(torch.load(filepath_4, map_location=device))

    model.eval()
    model_2.eval()
    model_3.eval()
    model_4.eval()

    # val_weights = validation_weights([model, model_2, model_3, model_4], dataset_val)
    val_weights = [0.5279900411868104, 0.04605057596583206, 0.11095235215754355, 0.31500703068981406]

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
    results_mobilev3 = pd.DataFrame(columns=columns)
    results_densenet121 = pd.DataFrame(columns=columns)
    results_resnet50_nms = pd.DataFrame(columns=columns)
    results_resnet18_nms = pd.DataFrame(columns=columns)
    results_mobilev3_nms = pd.DataFrame(columns=columns)
    results_densenet121_nms = pd.DataFrame(columns=columns)
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
            prediction_3 = model_3([img.to(device)])[0]
            prediction_4 = model_4([img.to(device)])[0]

        ### VISUALISE MODELS PREDICTIONS
        metrics_50 = visualise_model_prediction(prediction, target, img, image_id, save, show, path_prediction, 'RESNET50')
        metrics_18 = visualise_model_prediction(prediction_2, target, img, image_id, save, show, path_prediction, 'RESNET18')
        metrics_v3 = visualise_model_prediction(prediction_3, target, img, image_id, save, show, path_prediction, 'mobilenet_v3')
        metrics_121 = visualise_model_prediction(prediction_4, target, img, image_id, save, show, path_prediction, 'densenet_121')

        ### VISUALISE MODELS PREDICTIONS AFTER IOU_THERESHOLD
        nms_prediction, metrics_50_nms = visualise_model_prediction_nms(prediction, target, img, image_id, save, show, path_prediction, 'RESNET50_NMS')
        nms_prediction_2, metrics_18_nms = visualise_model_prediction_nms(prediction_2, target, img, image_id, save, show, path_prediction, 'RESNET18_NMS')
        nms_prediction_3, metrics_v3_nms = visualise_model_prediction_nms(prediction_3, target, img, image_id, save, show, path_prediction, 'mobilenet_v3_NMS')
        nms_prediction_4, metrics_121_nms = visualise_model_prediction_nms(prediction_4, target, img, image_id, save, show, path_prediction,'densenet_121_NMS')

        all_prediction = [nms_prediction, nms_prediction_2, nms_prediction_3, nms_prediction_4]

        ### VISUALISE MODELS PREDICTIONS AFTER ENSEMBLING
        all_prediction = filtering_ensemble(all_prediction, val_weights, image_id)
        all_val_weights = val_weights.copy()

        for i, pred in enumerate(all_prediction):
            plot_img_bbox(img, pred, title=f'{i}_filter', save=save,
                          image_id=image_id, show=show, path=path_prediction)

        while len(all_prediction) != 1:
            inter_val_weights = []
            nms_pred = all_prediction.pop()
            nms_pred_2 = all_prediction.pop()
            inter_val_weights.append(all_val_weights.pop())
            inter_val_weights.append(all_val_weights.pop())
            compose_bbox = ensemble_OD_predictions([nms_pred['boxes'], nms_pred_2['boxes']],
                                                   [nms_pred['labels'], nms_pred_2['labels']],
                                                   [nms_pred['scores'], nms_pred_2['scores']],
                                                   weights=inter_val_weights)
            all_prediction.append(compose_bbox)
            all_val_weights.append(max(inter_val_weights))
        compose_bbox = all_prediction[0]

        plot_img_bbox(img, compose_bbox, title='ENSEMBLE', save=save,
                      image_id=image_id, show=show, path=path_prediction)
        metrics_ensemble_nms = calculate_coco_metrics(target, compose_bbox)

        ### SAVE METRICS FOR CURRENT IMAGE
        all_results = [results_resnet50, results_resnet18, results_mobilev3, results_densenet121, results_resnet50_nms,
                       results_resnet18_nms, results_mobilev3_nms, results_densenet121_nms, results_ensemble_nms]
        all_metrics = [metrics_50, metrics_18, metrics_v3, metrics_121, metrics_50_nms,
                       metrics_18_nms, metrics_v3_nms, metrics_121_nms, metrics_ensemble_nms]
        all_names = ['resnet50', 'resnet18', 'mobilenet_v3', 'densenet_121', 'resnet50_nms',
                     'resnet18_nms', 'mobilenet_v3_nms', 'densenet_121', 'ensemble']

        for res, met in zip(all_results, all_metrics):
            res = res.append(met, ignore_index=True)

        for name, met in zip(all_names, all_metrics):
            result_current_image[name] = list(met.values())

        result_current_image.index = columns
        result_current_image.to_csv(os.path.join(path_prediction, image_id, f'{image_id}.csv'))

        # TODO apply_nms after ensemble

    ### SAVE CSV FILES
    for res in all_results:
        res['image_id'] = image_ids

    for res, name in zip(all_results, all_names):
        res.to_csv(os.path.join(path_prediction, f'{name}.csv'))

    gc.collect()
