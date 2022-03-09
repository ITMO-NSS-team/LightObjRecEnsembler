import numpy as np
import torch
from PIL import Image
import json
from tqdm import tqdm
from os.path import join, isfile
from torchvision import transforms
from baseline_VHR.data_loaders.data_constants import xView_PATH_TO_IMAGES, xView_PATH_TO_JSON


class xViewDataset(torch.utils.data.Dataset):
    """
    Class-dataloader for xView dataset (https://challenge.xviewdataset.org/download-links)

    
    """
    classes = []

    def __init__(self):
        self.imgs, self.boxes, self.classes = self._read_xView(xView_PATH_TO_IMAGES, xView_PATH_TO_JSON)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        num_objs = len(self.boxes[idx])
        boxes = torch.as_tensor(self.boxes[idx], dtype=torch.float32)
        labels = torch.as_tensor(self.classes[idx], dtype=torch.int64)
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
        transf = transforms.ToTensor()
        img = transf(img)
        return img, target

    def __len__(self):
        return len(self.imgs)


    def _is_more_than_zero(self, box: list) -> bool:
        """
        Check that all numbers in the box are more than zero
        """
        out = True
        for item in box:
            if item < 0:
                out = False
        return out

    def _is_in_bounds(self, box_list, w, h) -> bool:
        output = True
        for box in box_list:
            if not (box[0] >= 0 and box[1] >= 0 and box[2] < w and box[3] < h):
                output = False
        return output

    def _read_xView(self, images_path: str, json_path: str) -> list:
        """
        This method reads images and annotations in xView format and returns list of pairs

        :param images_path - path to images folder
        :param json_path - path to annotations file

        :return pair_list - list of data pairs
        """
        list_of_classes = []
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
                if not class_of_object in list_of_classes:
                    list_of_classes.append(class_of_object)
                if object_bb.shape != (4,):
                    print("Issues at %d!" % i)
                    return [], [], []
                if image == "":
                    image = image_name
                    objects = []
                    classes = []
                    if self._is_more_than_zero(object_bb):
                        objects.append(object_bb)
                        classes.append(class_of_object)
                elif image == image_name:
                    if self._is_more_than_zero(object_bb):
                        objects.append(object_bb)
                        classes.append(class_of_object)
                elif image != image_name:
                    if (isfile(join(images_path, image))):
                        img = Image.open(join(images_path, image)).convert("RGB")
                        img_width, img_height = img.size
                        if self._is_in_bounds(objects, img_width, img_height):
                            if len(objects):
                                out_images.append(join(images_path, image))
                                out_boxes.append(objects)
                                out_classes.append(classes)
                    image = image_name
                    objects = []
                    classes = []
                    if self._is_more_than_zero(object_bb):
                        objects.append(object_bb)
                        classes.append(class_of_object)
        
        if (isfile(join(images_path, image))):
            img = Image.open(join(images_path, image)).convert("RGB")
            img_width, img_height = img.size
            if self._is_in_bounds(objects, img_width, img_height):
                if len(objects):
                    out_images.append(join(images_path, image))
                    out_boxes.append(objects)
                    out_classes.append(classes)

        list_of_classes.sort()
        print(list_of_classes)
        for i in range(len(out_classes)):
            for j in range(len(out_classes[i])):
                out_classes[i][j] = list_of_classes.index(out_classes[i][j])
        #print(list_of_classes)
        print(len(list_of_classes))
        return out_images, out_boxes, out_classes