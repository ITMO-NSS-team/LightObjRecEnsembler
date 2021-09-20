import itertools
import pandas as pd
from models import Darknet
from utils import utils
import time, datetime, random
import numpy as np
import torch
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from fast_rcnn.utils.config import opt
from fast_rcnn.model.faster_rcnn_vgg16 import FasterRCNNVGG16
from fast_rcnn.trainer import FasterRCNNTrainer
from fast_rcnn.data.util import read_image
from fast_rcnn.utils.vis_tool import vis_bbox
from fast_rcnn.utils import array_tool as at
from baseline_VHR.utils.ensemble import Rectangle

Tensor = torch.FloatTensor
img_path = "images/Intersection-Counts.jpg"

rcnn_label = ['fly', 'bike', 'bird', 'boat', 'pin',
              'bus', 'c', 'cat', 'chair', 'cow', 'table',
              'dog', 'horse', 'moto',
              'p', 'plant', 'shep', 'sofa', 'train', 'tv', 'bg']

YOLO_label = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
              'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
              'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
              'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
              'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
              'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
              'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable',
              'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
              'teddy bear', 'hair drier', 'toothbrush']


def create_model(img_size):
    config_path = 'config/yolov3.cfg'
    weights_path = 'config/yolov3.weights'
    class_path = 'config/coco.names'

    # Load model and weights
    model = Darknet(config_path, img_size=img_size)
    model.load_weights(weights_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # model.cuda()
    model.eval()
    classes = utils.load_classes(class_path)
    return model, classes


def detect_image(img,
                 model,
                 img_size,
                 conf_thres,
                 nms_thres):
    #scale and pad image
    ratio = min(img_size / img.size[0], img_size / img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    #imw = img.size[0]
    #imh = img.size[1]
    img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
                                         transforms.Pad((max(int((imh - imw) / 2), 0),
                                                         max(int((imw - imh) / 2), 0), max(int((imh - imw) / 2), 0),
                                                         max(int((imw - imh) / 2), 0)), (128, 128, 128)),
                                         transforms.ToTensor(),
                                         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    #input_img = torch.from_numpy(img)[None]
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80,
                                               conf_thres, nms_thres)
    return detections[0]


def get_predictions(img_path,
                    model,
                    img_size,
                    conf_thres,
                    nms_thres
                    ):
    prev_time = time.time()
    img = Image.open(img_path)
    #img = read_image(img_path)
    detections = detect_image(img,
                              model,
                              img_size,
                              conf_thres,
                              nms_thres
                              )
    inference_time = datetime.timedelta(seconds=time.time() - prev_time)
    print('Inference Time: %s' % (inference_time))
    return img, detections


def plot_predictions(img,
                     detections,
                     classes,
                     img_size,
                     convert_labels_flag=False):
    # Get bounding-box colors
    cmap = plt.get_cmap('tab20b')
    unpadded_bbox = []
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    img = np.array(img)
    plt.figure()
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    if convert_labels_flag:
        unique_labels = set(detections[1])
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for bbox, cls_pred in zip(detections[0], detections[1]):
            x1, y1, x2, y2 = bbox
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
            y2 = ((y2 - pad_y // 2) / unpad_h) * img.shape[0]
            x2 = ((x2 - pad_x // 2) / unpad_w) * img.shape[1]
            bbox_unpadded = x1.item(), y1.item(), x2.item(), y2.item()
            unpadded_bbox.append(bbox_unpadded)
            try:
                color = bbox_colors[int(np.where(
                    unique_labels == int(cls_pred))[0])]
            except Exception as ex:
                ex = 1
                color = bbox_colors[0]
            bbox = patches.Rectangle((x1, y1), box_w, box_h,
                                     linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(bbox)
            if convert_labels_flag:
                text = str(cls_pred)
            else:
                text = classes[int(cls_pred)]
            plt.text(x1, y1, s=text,
                     color='white', verticalalignment='top',
                     bbox={'color': color, 'pad': 0})

        plt.axis('off')
        # save image
        plt.savefig(img_path.replace(".jpg", "-det.jpg"),
                    bbox_inches='tight', pad_inches=0.0)
        plt.show()
        return unpadded_bbox
    else:
        if detections is not None:
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            # browse detections and draw bounding boxes
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
                y2 = ((y2 - pad_y // 2) / unpad_h) * img.shape[0]
                x2 = ((x2 - pad_x // 2) / unpad_w) * img.shape[1]
                #300.9246, 279.0023, 315.5650, 320.5440
                bbox_unpadded = x1.item(), y1.item(), x2.item(), y2.item()
                unpadded_bbox.append(bbox_unpadded)
                color = bbox_colors[int(np.where(
                    unique_labels == int(cls_pred))[0])]
                bbox = patches.Rectangle((x1, y1), box_w, box_h,
                                         linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(bbox)
                if convert_labels_flag:
                    text = str(cls_pred)
                else:
                    text = classes[int(cls_pred)]
                plt.text(x1, y1, s=text,
                         color='white', verticalalignment='top',
                         bbox={'color': color, 'pad': 0})
        plt.axis('off')
        # save image
        plt.savefig(img_path.replace(".jpg", "-det.jpg"),
                    bbox_inches='tight', pad_inches=0.0)
        plt.show()
        return unpadded_bbox

def convert_predictions(predictions):
    bboxes = [x[:4].tolist() for x in predictions]
    scores = [x[5].item() for x in predictions]
    labels = [x[6].item() for x in predictions]
    return bboxes, scores, labels


def YOLO_branch(img_path: str,
                conf_thres: float = 0.8,
                nms_thres: float = 0.4,
                img_size: int = 416,
                vis: bool = False):
    model, classes = create_model(img_size)
    img, detections = get_predictions(img_path, model, img_size, conf_thres, nms_thres)
    converted_labels = []

    bboxes, scores, labels = convert_predictions(detections)

    for label_index in labels:
        converted_labels.append(YOLO_label[int(label_index)])

    if vis:
        bboxes = plot_predictions(img, detections, classes, img_size)

    return bboxes, converted_labels, scores, classes, img


def fast_RCNN_branch(img_path: str,
                     conf_thres: float = 0.8,
                     nms_thres: float = 0.4,
                     img_size: int = 3600,
                     vis: bool = False):
    rcnn_flag = True

    faster_rcnn = FasterRCNNVGG16()
    cuda_flag = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = FasterRCNNTrainer(faster_rcnn).to(device)
    trainer.load('./config/RCNN_weights.pth')

    if rcnn_flag:
        img = read_image(img_path)
        input_img = torch.from_numpy(img)[None]
        bboxes, labels, scores = trainer.faster_rcnn.predict(input_img, visualize=True)
    else:
        img = Image.open(img_path)
        ratio = min(img_size / img.size[0], img_size / img.size[1])
        imw = round(img.size[0] * ratio)
        imh = round(img.size[1] * ratio)
        img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
                                             transforms.Pad((max(int((imh - imw) / 2), 0),
                                                             max(int((imw - imh) / 2), 0), max(int((imh - imw) / 2), 0),
                                                             max(int((imw - imh) / 2), 0)), (128, 128, 128)),
                                             transforms.ToTensor(),
                                             ])
        # convert image to Tensor
        image_tensor = img_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input_img = Variable(image_tensor.type(Tensor))
        opt.caffe_pretrain = False  # this model was trained from torchvision-pretrained model
        bboxes, labels, scores = trainer.faster_rcnn.predict(input_img, visualize=True)

    if vis:
        vis_bbox(at.tonumpy(input_img[0]),
                 at.tonumpy(bboxes[0]),
                 at.tonumpy(labels[0]).reshape(-1),
                 at.tonumpy(scores[0]).reshape(-1))

    converted_labels = []
    for label_index in labels[0]:
        converted_labels.append(rcnn_label[label_index])

    return bboxes[0], converted_labels, scores


def rectangle_intersect(first_box: list,
                        second_box: list):
    a = Rectangle(*first_box)
    b = Rectangle(*second_box)
    area = a & b
    if area is None:
        return area, None
    else:
        intersect_area = area.calculate_area()
        #composite_bbox = a - area
        composite_bbox = a.difference(area)
        ratio_1, ratio_2 = intersect_area / a.calculate_area(), intersect_area / b.calculate_area()
        return (ratio_1, ratio_2), composite_bbox



def ensemble_OD_predictions(bboxes: list,
                            labels: list,
                            scores: list,
                            image: np.ndarray,
                            img_size: int = 416,
                            area_threshold: float = 0.35,
                            ensemble_type: str = 'Majority',
                            vis_flag: bool = True):
    if len(bboxes[0]) > len(bboxes[1]):
        bboxes.reverse()

    #bboxes[0] = bboxes[0].tolist()
    ratio_list = []
    image_list = []
    intersect_coord_list = []
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
        for i in range(0, len(first_box) - 1, 2):
            first_box[i], first_box[i + 1] = first_box[i + 1], first_box[i]
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

    # composite_bbox, composite_label = majority_ensemble()

    if vis_flag:
        plot_predictions(image,
                         [list(itertools.chain(bboxes[1], bboxes_merged)),
                          list(itertools.chain(labels[1], labels_merged))],
                         classes,
                         img_size,
                         convert_labels_flag=True)
    tmp = 1

    return bboxes_merged


# load image and get detections
if __name__ == '__main__':
    f = Image.open(img_path)
    bboxes1, labels1, scores1, classes, img = YOLO_branch(img_path=img_path, vis=True)
    bboxes, labels, scores = fast_RCNN_branch(img_path=img_path, vis=True)
    ratio_1, ratio_2 = ensemble_OD_predictions([bboxes, bboxes1],
                                               [labels, labels1],
                                               [scores, scores1],
                                               image=img)
    print('ttt')
