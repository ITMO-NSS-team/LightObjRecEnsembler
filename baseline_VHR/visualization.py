import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_img_bbox(img, target, image_id: str = None, title: str = None,
                  save: bool = False, path: str = None, show: bool = False):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)

    if title:
        fig.suptitle(title, fontsize=20)

    im = img.cpu().detach().numpy()
    im = im.transpose((1, 2, 0))
    a.imshow(im)
    for box in (target['boxes']):
        x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth=2,
                                 edgecolor='r',
                                 facecolor='none')

        # Draw the bounding box on top of the image
        a.add_patch(rect)

    if save:
        if path is None:
            path = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(path, 'NWPU VHR-10 dataset', 'predictions')
        dirpath = os.path.join(path, image_id)

        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        filepath = os.path.join(dirpath, f'{title}.png')
        fig.savefig(filepath)

    if show:
        plt.show()
