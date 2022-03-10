from __future__ import division
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pathlib
import numpy as np
import torch
from typing import List
#from datasets import ObjectDetectionDataSet
from torchvision.models.detection.transform import GeneralizedRCNNTransform

#import matplotlib.pyplot as plt
#import matplotlib.patches as patches


def from_file_to_BoundingBox(file_name: pathlib.Path, groundtruth: bool = True):
    """Returns a list of BoundingBox objects from groundtruth or prediction."""
    from baseline_utils.metrics.general_utils import BoundingBox
    from baseline_utils.metrics.enumerators import BBFormat, BBType

    file = torch.load(file_name)
    labels = file['labels']
    boxes = file['boxes']
    scores = file['scores'] if not groundtruth else [None] * len(boxes)

    gt = BBType.GROUND_TRUTH if groundtruth else BBType.DETECTED

    return [BoundingBox(image_name=file_name.stem,
                        class_id=l,
                        coordinates=tuple(bb),
                        format=BBFormat.XYX2Y2,
                        bb_type=gt,
                        confidence=s) for bb, l, s in zip(boxes, labels, scores)]


def from_dict_to_BoundingBox(file: dict, name: str, groundtruth: bool = True):
    """Returns list of BoundingBox objects from groundtruth or prediction."""
    from baseline_utils.metrics import BoundingBox
    from baseline_utils.metrics.enumerators import BBFormat, BBType

    labels = file['labels']
    boxes = file['boxes']
    scores = np.array(file['scores'].cpu()) if not groundtruth else [None] * len(boxes)

    gt = BBType.GROUND_TRUTH if groundtruth else BBType.DETECTED

    return [BoundingBox(image_name=name,
                        class_id=int(l),
                        coordinates=tuple(bb),
                        format=BBFormat.XYX2Y2,
                        bb_type=gt,
                        confidence=s) for bb, l, s in zip(boxes, labels, scores)]


