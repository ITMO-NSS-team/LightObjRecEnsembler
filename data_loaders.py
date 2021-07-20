import os
import numpy as np
import torch
from PIL import Image
import torch_utils.transforms as T
import torch_utils.utils as utils

def train_test_split(dataset_class):
    dataset = dataset_class('NWPU VHR-10 dataset', get_transform(train=False))
    dataset_test = dataset_class('NWPU VHR-10 dataset', get_transform(train=False))
    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()

    # train test split
    test_split = 0.2
    tsize = int(len(dataset)*test_split)
    dataset = torch.utils.data.Subset(dataset, indices[:-tsize])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-tsize:])
    return dataset, dataset_test

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

class VHRDataset(torch.utils.data.Dataset):
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
                    symbols = ["(", ")","\n"," "]
                    for symbol in symbols:
                        line = line.replace(symbol,"")
                    line = np.array(line.split(',')).astype(np.int64)
                    boxes.append(list(line[:4]))
                    labels.append(line[len(line)-1])
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