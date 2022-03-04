import os
import numpy as np
import torch
from PIL import Image
import torch_utils.transforms as T
import json
from tqdm import tqdm
from os.path import join, isfile

def train_test_split(dataset_class, validation_flag: bool = False):
    dataset = dataset_class('./NWPU VHR-10 dataset', get_transform(train=False))
    dataset_test = dataset_class('./NWPU VHR-10 dataset', get_transform(train=False))
    dataset_validation = dataset_class('./NWPU VHR-10 dataset', get_transform(train=False))
    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()

    # train test split
    test_split = 0.2
    valida_split = 0.2
    tsize = int(len(dataset) * test_split)
    vsize = int(len(dataset) * test_split * valida_split)
    dataset = torch.utils.data.Subset(dataset, indices[:-tsize])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-tsize:-vsize])
    if validation_flag:
        dataset_validation = torch.utils.data.Subset(dataset_validation, indices[-vsize:])
    else:
        dataset_validation = None

    return dataset, dataset_test, dataset_validation


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

class xViewDataset(torch.utils.data.Dataset):

    classes = []
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        path_to_image_folder = ""
        path_to_json_file = ""
        self.imgs, self.boxes, self.classes = read_xView(path_to_image_folder, path_to_json_file)

        #self.imgs = list(sorted(os.listdir(os.path.join(self.root, "positive image set"))))
        #self.boxes = list(sorted(os.listdir(os.path.join(self.root, "ground truth"))))

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        num_objs = len(self.boxes[idx])
        boxes = torch.as_tensor(self.boxes, dtype=torch.float32)
        labels = torch.as_tensor(self.classes, dtype=torch.int64)
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

def _is_more_than_zero(box: list) -> bool:
    out = True
    for item in box:
        if item < 0:
            out = False
    return out


def read_xView(images_path: str, json_path: str) -> list:
    """
    This method reads images and annotations in xView format and returns list of pairs

    :param images_path - path to images folder
    :param json_path - path to annotations file

    :return pair_list - list of data pairs
    """
    out_images = []
    out_boxes = []
    out_classes = []
    with open(json_path) as f:
        data = json.load(f)
    objects = []
    image = ""
    classes = []
    for i in tqdm(range(len(data['features']))):
        if data['features'][i]['properties']['bounds_imcoords'] != []:
            image_name = data['features'][i]['properties']['image_id']
            object_bb = np.array([int(num) for num in data['features'][i]['properties']['bounds_imcoords'].split(",")])
            class_of_object = data['features'][i]['properties']['type_id']
            if object_bb.shape[0] != 4:
                print("Issues at %d!" % i)
                return [], [], []
            if image == "":
                image = image_name
                objects = []
                classes = []
                if _is_more_than_zero(object_bb):
                    objects.append(object_bb)
                    classes.append(class_of_object)
            elif image == image_name:
                if _is_more_than_zero(object_bb):
                    objects.append(object_bb)
                    classes.append(class_of_object)
            elif image != image_name:
                if (isfile(join(images_path, image))):
                    out_images.append(join(images_path, image))
                    out_boxes.append(objects)
                    out_classes.append(out_classes)
                image = image_name
                objects = []
                classes = []
                if _is_more_than_zero(object_bb):
                    objects.append(object_bb)
                    classes.append(class_of_object)
    if (isfile(join(images_path, image))):
        out_images.append(join(images_path, image))
        out_boxes.append(objects)
        out_classes.append(out_classes)
    return out_images, out_boxes, out_classes