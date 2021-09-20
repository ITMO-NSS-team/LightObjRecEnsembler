from copy import deepcopy
import gc
import itertools

import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import baseline_VHR.torch_utils.transforms as T
from baseline_VHR.torch_utils.engine import train_one_epoch, evaluate
import baseline_VHR.torch_utils.utils as utils
from baseline_VHR.data_loaders import train_test_split, VHRDataset
from baseline_VHR.visualization import plot_img_bbox
from baseline_VHR.faster_RCNN_baseline import get_fasterRCNN_resnet
from baseline_VHR.utils.ensemble import Rectangle


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def apply_nms(orig_prediction, iou_thresh=0.3):
    # torchvision returns the indices of the boxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = deepcopy(orig_prediction)
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction


def torch_to_pil(img):
    return transforms.ToPILImage()(img).convert('RGB')


def train_model(model, device, dataset, dataset_test, num_epochs=10):
    gc.collect()
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=10, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=10, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    # model = get_object_detection_model(num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(num_epochs):
        # training for one epoch
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
    path = "./VHR_resnet18.pth"
    filepath = "./VHR_statedict_resnet18.pth"
    torch.save(model, path)
    torch.save(model.state_dict(), filepath)


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
                            img_size: int = 416,
                            area_threshold: float = 0.75,
                            ensemble_type: str = 'Majority',
                            vis_flag: bool = True):
    if len(bboxes[0]) > len(bboxes[1]):
        bboxes.reverse()
        labels.reverse()
        scores.reverse()

    bboxes_merged = []
    labels_merged = []

    mergedDf = pd.DataFrame({'Weak NN frame number': [],
                             'Weak NN frame class': [],
                             'Major NN frame number': [],
                             'Major NN frame class': [],
                             'Intersection area/Weak NN area': [],
                             'Intersection area/Major NN area': [],
                             'New frame coordinates': []},
                            columns=['Weak NN frame number',
                                     'Weak NN frame class',
                                     'Major NN frame number',
                                     'Major NN frame class',
                                     'Intersection area/Weak NN area',
                                     'Intersection area/Major NN area',
                                     'New frame coordinates'])

    for index_1, first_box in enumerate(bboxes[0]):
        for index_2, second_box in enumerate(bboxes[1]):
            ratio, intersect_coord = rectangle_intersect(first_box, second_box)
            if ratio is None:
                check_flag = True
            elif sum(ratio) > area_threshold:
                mergedDf = mergedDf.append({'Weak NN frame number': index_1,
                                            'Weak NN frame class': labels[0][index_1],
                                            'Major NN frame number': index_2,
                                            'Major NN frame class': labels[1][index_2],
                                            'Intersection area/Weak NN area': ratio[0],
                                            'Intersection area/Major NN area': ratio[1],
                                            'New frame coordinates': intersect_coord},
                                           ignore_index=True)

                check_flag = False
                break

        if check_flag:
            bboxes_merged.append(first_box)
            labels_merged.append(labels[0][index_1])

    compose_bbox = list(itertools.chain(bboxes[1], bboxes_merged)),
    compose_labels = list(itertools.chain(labels[1], labels_merged))

    return compose_bbox[0]


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

train_mode = False
dataset, dataset_test = train_test_split(VHRDataset)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
filepath = "../local/VHR_statedict_fasterrcnn_resnet50_fpn.pth"
filepath_2 = "../local/VHR_statedict_resnet18.pth"

if __name__ == '__main__':
    if train_mode:
        train_model(model, device, dataset, dataset_test, num_epochs=15)
    else:
        model.load_state_dict(torch.load(filepath))
        model_2.load_state_dict(torch.load(filepath_2))

    img, target = dataset_test[27]
    plot_img_bbox(img, target)

    model.eval()
    model_2.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])[0]
        prediction_2 = model_2([img.to(device)])[0]

    print('predicted model #boxes: ', len(prediction['labels']))
    print('predicted model_2 #boxes: ', len(prediction_2['labels']))
    print('real #boxes: ', len(target['labels']))

    print('MODEL OUTPUT')

    pred = deepcopy(prediction)
    pred['boxes'] = pred['boxes'].cpu().numpy()
    plot_img_bbox(img, pred)

    pred = deepcopy(prediction_2)
    pred['boxes'] = pred['boxes'].cpu().numpy()
    plot_img_bbox(img, pred)

    nms_prediction = apply_nms(prediction, iou_thresh=0.05)
    nms_prediction_2 = apply_nms(prediction_2, iou_thresh=0.05)

    print('NMS APPLIED MODEL OUTPUT')
    nms_prediction['boxes'] = nms_prediction['boxes'].cpu().numpy()
    nms_prediction_2['boxes'] = nms_prediction_2['boxes'].cpu().numpy()
    nms_prediction['labels'] = nms_prediction['labels'].cpu().numpy()
    nms_prediction_2['labels'] = nms_prediction_2['labels'].cpu().numpy()
    nms_prediction['scores'] = nms_prediction['scores'].cpu().numpy()
    nms_prediction_2['scores'] = nms_prediction_2['scores'].cpu().numpy()
    plot_img_bbox(img, nms_prediction)
    plot_img_bbox(img, nms_prediction_2)
    print('NMS APPLIED MODEL OUTPUT')

    compose_bbox = ensemble_OD_predictions([nms_prediction_2['boxes'], nms_prediction['boxes']],
                                           [nms_prediction_2['labels'], nms_prediction['labels']],
                                           [nms_prediction_2['scores'], nms_prediction['boxes']],
                                           image=img)

    plot_img_bbox(img, {'boxes': compose_bbox})

    gc.collect()
