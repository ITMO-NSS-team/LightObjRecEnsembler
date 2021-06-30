from models import Darknet
from utils import utils
import os, sys, time, datetime, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from fast_rcnn.utils.config import opt
from fast_rcnn.model.faster_rcnn_vgg16 import FasterRCNNVGG16
from fast_rcnn.trainer import FasterRCNNTrainer
from fast_rcnn.data.util import  read_image
from fast_rcnn.utils.vis_tool import vis_bbox
from fast_rcnn.utils import array_tool as at

Tensor = torch.FloatTensor
img_path = "images/blueangels.jpg"



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
    # scale and pad image
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
                     img_size):
    # Get bounding-box colors
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    img = np.array(img)
    plt.figure()
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
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
            color = bbox_colors[int(np.where(
                unique_labels == int(cls_pred))[0])]
            bbox = patches.Rectangle((x1, y1), box_w, box_h,
                                     linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(bbox)
            plt.text(x1, y1, s=classes[int(cls_pred)],
                     color='white', verticalalignment='top',
                     bbox={'color': color, 'pad': 0})
    plt.axis('off')
    # save image
    plt.savefig(img_path.replace(".jpg", "-det.jpg"),
                bbox_inches='tight', pad_inches=0.0)
    plt.show()


def YOLO_branch(img_path: str,
                conf_thres: float = 0.8,
                nms_thres: float = 0.4,
                img_size: int = 416,
                vis: bool = False):
    model, classes = create_model(img_size)
    img, detections = get_predictions(img_path, model, img_size, conf_thres, nms_thres)
    if vis:
        plot_predictions(img, detections, classes, img_size)
    return img, detections


def fast_RCNN_branch(img_path: str,
                     conf_thres: float = 0.8,
                     nms_thres: float = 0.4,
                     img_size: int = 416,
                     vis: bool = False):
    img = read_image(img_path)
    img = torch.from_numpy(img)[None]
    faster_rcnn = FasterRCNNVGG16()
    cuda_flag = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = FasterRCNNTrainer(faster_rcnn).to(device)
    trainer.load('./config/RCNN_weights.pth')
    opt.caffe_pretrain = False  # this model was trained from torchvision-pretrained model
    bboxes, labels, scores = trainer.faster_rcnn.predict(img, visualize=True)

    if vis:
        vis_bbox(at.tonumpy(img[0]),
                 at.tonumpy(bboxes[0]),
                 at.tonumpy(labels[0]).reshape(-1),
                 at.tonumpy(scores[0]).reshape(-1))

    return bboxes, labels, scores


# load image and get detections
if __name__ == '__main__':
    bboxes, labels, scores = fast_RCNN_branch(img_path=img_path, vis=True)
    img1, detections1 = YOLO_branch(img_path=img_path, vis=True)
