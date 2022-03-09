import numpy as np
import torch
import json
from PIL import Image
from tqdm import tqdm
from os.path import join, isfile
from torchvision import transforms
from baseline_VHR.data_loaders.data_constants import METU_PATH_TO_TRAIN_IMAGES, METU_PATH_TO_TRAIN_JSON
from baseline_VHR.data_loaders.data_constants import METU_PATH_TO_TEST_IMAGES, METU_PATH_TO_TEST_JSON
from baseline_VHR.data_loaders.data_constants import METU_PATH_TO_VAL_IMAGES, METU_PATH_TO_VAL_JSON
from baseline_VHR.data_loaders.data_constants import METU_TEST, METU_VAL, METU_TRAIN

class METUDataset(torch.utils.data.Dataset):
    classes = []

    def __init__(self,
                 val_flag: int = METU_TRAIN):
        if val_flag == METU_TRAIN:
            path_to_image_folder = METU_PATH_TO_TRAIN_IMAGES
            path_to_json_file = METU_PATH_TO_TRAIN_JSON
        elif val_flag == METU_VAL:
            path_to_image_folder = METU_PATH_TO_VAL_IMAGES
            path_to_json_file = METU_PATH_TO_VAL_JSON
        elif val_flag == METU_TEST:
            path_to_image_folder = METU_PATH_TO_TEST_IMAGES
            path_to_json_file = METU_PATH_TO_TEST_JSON
        else:
            print ("METU tag error! Check data_constants for tags")
            return 1
        self.imgs, self.boxes, self.classes = self._read_METU(path_to_image_folder, path_to_json_file)

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

    def _is_more_than_zero(box: list) -> bool:
        out = True
        for item in box:
            if item < 0:
                out = False
        return out

    def _search_filename_by_imageID(data, imageID: int) -> str:
        for i in range(len(data['images'])):
            if data['images'][i]['id'] == imageID:
                return data['images'][i]['file_name']


    def _read_METU(self, images_path: str, json_path: str) -> list:
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
        for i in tqdm(range(len(data['annotations']))):
            if data['annotations'][i]['bbox'] != []:
                image_id = data['annotations'][i]['image_id']
                object_bb_1 = np.array([int(num) for num in data['annotations'][i]['bbox']])
                object_bb = np.array(([object_bb_1[0], object_bb_1[1], object_bb_1[0]+object_bb_1[2], object_bb_1[1]+object_bb_1[3]]))
                class_of_object = data['annotations'][i]['category_id']
                if object_bb.shape[0] != 4:
                    print("Issues at %d!" % i)
                    return out_images, out_boxes, out_classes
                if image == "":
                    image = image_id
                    objects = []
                    classes = []
                    if self._is_more_than_zero(object_bb):
                        objects.append(object_bb)
                        classes.append(class_of_object)
                elif image == image_id:
                    if self._is_more_than_zero(object_bb):
                        objects.append(object_bb)
                        classes.append(class_of_object)
                elif image != image_id:
                    image_name = self._search_filename_by_imageID(data, image)
                    if (isfile(join(images_path, image_name))):
                        out_images.append(join(images_path, image_name))
                        out_boxes.append(objects)
                        out_classes.append(classes)
                    image = image_id
                    objects = []
                    classes = []
                    if self._is_more_than_zero(object_bb):
                        objects.append(object_bb)
                        classes.append(class_of_object)
                    
            image_name = self._search_filename_by_imageID(data, image)
            if (isfile(join(images_path, image_name))):
                        out_images.append(join(images_path, image_name))
                        out_boxes.append(objects)
                        out_classes.append(classes)
        return out_images, out_boxes, out_classes
