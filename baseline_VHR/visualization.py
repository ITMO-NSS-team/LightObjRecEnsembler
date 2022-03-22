from cProfile import label
import os
from PIL import Image
from unicodedata import name
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import numpy


def plot_img_bbox(img, target, device, classes, image_id: str = None, title: str = None,
                  save: bool = False, path: str = None, show: bool = False):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height

    dirpath = os.path.join(path, image_id)

    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    if save or show:
        fig, a = plt.subplots(1, 1)
        fig.set_size_inches(5, 5)

        if title:
            fig.suptitle(title, fontsize=20)

        im = img.to(device).detach().numpy()
        im = im.transpose((1, 2, 0))
        a.imshow(im)
        plt.close('all')
        for box, labl in zip(target['boxes'], target['labels']):
            x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
            rect = patches.Rectangle((x, y),
                                     width, height,
                                     linewidth=2,
                                     edgecolor='r',
                                     facecolor='none')
            #rect = rect.numpy()
            # Draw the bounding box on top of the image
            a.add_patch(rect)
            rx, ry = rect.get_xy()
            cx = int(rx + rect.get_width()/2.0)
            cy = int(ry + rect.get_height()/2.0)
            if len(target['boxes']) < 10:
                a.annotate(classes[int(labl.item()-1)], (cx, cy), color='red', fontsize=10, ha='center', va='center')
            else:
                a.annotate(str(int(labl.item()-1)), (cx, cy), color='red', fontsize=10, ha='center', va='center')
        plt.close('all')

    if save:
        filepath = os.path.join(dirpath, f'{title}.png')
        fig.savefig(filepath)

    if show:
        plt.show()


def plot_img_bbox_all(img, target, device, classes, image_id: str = None, title: list = ["clear", 'fasterrcnn_resnet50_fpn', 
                                                                                        'fasterrcnn_resnet50_fpn_NMS','resnet18', 'resnet18_NMS',
                                                                                'mobilenet_v3_large','mobilenet_v3_large_NMS','densenet121', 
                                                                                'densenet121_NMS', "ensamble"],
                  save: bool = False, path: str = None, show: bool = False):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height

    dirpath = os.path.join(path, image_id)

    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    if save or show:
        #fig, a = plt.subplots(1, 5)
        #fig.set_size_inches(5, 5)
        my_dpi = 30
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(70, 40), dpi=my_dpi)
        if title:
            fig.suptitle("All images", fontsize=70)
        counter = 0
        for i in range(2):
            for j in range(5):
                if counter < 10:
                    im = img.to(device).detach().numpy()
                    im = im.transpose((1, 2, 0))
                    axes[i, j].set_title(title[counter], fontsize = 50)

                    axes[i, j].imshow(im)
                    axes[i, j].axis('off')  
                    #arr[-1].set_size_inches(5, 5)
                    #a[i] = plt.imshow(im)
                    plt.close('all')
                    for box, labl in zip(target[counter]['boxes'], target[counter]['labels']):
                        x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
                        rect = patches.Rectangle((x, y),
                                                width, height,
                                                linewidth=10,
                                                edgecolor='r',
                                                facecolor='none')
                        #rect = rect.numpy()
                        # Draw the bounding box on top of the image
                        axes[i, j].add_patch(rect)
                        rx, ry = rect.get_xy()
                        cx = rx + rect.get_width()/2.0
                        cy = ry + rect.get_height()/2.0
                        size = 30
                        if len(target[counter]['boxes']) < 10:
                            axes[i, j].annotate(classes[labl.item()-1], (cx, cy), color='red', fontsize=50, ha='center', va='center')
                        else:
                            axes[i, j].annotate(str(labl.item()-1), (cx, cy), color='red', fontsize=50, ha='center', va='center')
                    plt.close('all') 
                    counter += 1
    if save:
        filepath = os.path.join(dirpath, f'all.png')
        fig.savefig(filepath)

    if show:
        plt.show()
