import os
import numpy as np
import torch
from PIL import Image
import json
from tqdm import tqdm
from os.path import join, isfile
from os.path import splitext
from torchvision import transforms

class VHRDataset(torch.utils.data.Dataset):
    """
    Work in progress
    """
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(self.root, "positive image set"))))
        self.boxes = list(sorted(os.listdir(os.path.join(self.root, "ground truth"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "positive image set", self.imgs[idx])
        box_path = os.path.join(self.root, "ground truth", self.boxes[idx])
        img = Image.open(img_path).convert("RGB")
        boxes = []
        labels = []
        with open(box_path) as f:
            for line in f:
                if "(" in line:
                    symbols = ["(", ")", "\n", " "]
                    for symbol in symbols:
                        line = line.replace(symbol, "")
                    line = np.array(line.split(',')).astype(np.int64)
                    boxes.append(list(line[:4]))
                    labels.append(line[len(line) - 1])
                else:
                    break
        num_objs = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)

