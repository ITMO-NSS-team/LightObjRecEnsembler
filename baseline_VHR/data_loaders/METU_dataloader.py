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
    """
    Class-loader for METU dataset.
    """
    classes = []

    def __init__(self, val_flag: int = METU_TRAIN):
        """
        :param val_flag is a flag for choosing of loading dataset. Can be METU_TEST, METU_VAL or METU_TRAIN. 
        Get them from baseline_VHR.data_loaders.data_constants
        """
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
        """
        Get item method of class. Can be an example for crething new dataloaders. Have to return list of images and list of targets
        :param idx - index of data pair 
        """
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
        """
        Returns length of dataset
        """
        return len(self.imgs)

    def _is_more_than_zero(self, box: list) -> bool:
        """
        Method-checker for boxes. Checks if all elements in boxes are >0
        :param box - [x1, y1, x2, y2]
        :return True if all elements are > 0 and False otherwise
        """
        out = True
        for item in box:
            if item <= 0:
                out = False
        return out
    
    def _is_not_degenerate(self, box: list) -> bool:
        """
        Method-checker for boxes. Checks if all elements in boxes are x1 < x2 and y1 < y2(not degenerate)
        :param box - [x1, y1, x2, y2]
        :return True if box isn't degenerate and False otherwise
        """
        if box[0] >= box[2] or box[1] >= box[3]:
            return False
        return True

    def _search_filename_by_imageID(self, data: dict, imageID: int) -> str:
        """
        Method-searcher for filenames in json annotation by imageID
        :param data - data read from json
        :param imageID - ide of image

        :return filename
        """
        for i in range(len(data['images'])):
            if data['images'][i]['id'] == imageID:
                return data['images'][i]['file_name']

    def _is_in_bounds(self, box_list, width, height) -> bool:
        """
        Method-checker for boxes. Checks if all elements in boxes are x1 >= 0, y1 >= 0, x1 < width and y2 < height 
        and that list of boxes isn't empty
        :param box_list - [[x1, y1, x2, y2], [x1, y1, x2, y2],...]
        :param width, height - params of image
        :return True all of boxes are om bounds and list of boxes isn't empty, False otherwise
        """
        output = True
        for box in box_list:
            if not (box[0] >= 0 and box[1] >= 0 and box[2] < width and box[3] < height):
                output = False
        if len(box_list) == 0:
            output = False
        return output

    def _read_METU(self, images_path: str, json_path: str) -> list:
        """
        This method reads images and annotations in xView format and returns lists of image, boxes and classes

        :param images_path - path to images folder
        :param json_path - path to annotations file

        :return out_images - list of images' filepaths
        :return out_boxes - list of boxes
        :return out_classess - list of classes
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
        for i in tqdm(range(len(data['annotations']))):
            if data['annotations'][i]['bbox'] != []:
                image_id = data['annotations'][i]['image_id']
                object_bb_1 = np.array([int(num) for num in data['annotations'][i]['bbox']])
                object_bb = np.array(([object_bb_1[0], object_bb_1[1], object_bb_1[0]+object_bb_1[2], object_bb_1[1]+object_bb_1[3]]))
                class_of_object = data['annotations'][i]['category_id']
                if not class_of_object in list_of_classes:
                    list_of_classes.append(class_of_object)
                if object_bb.shape != (4,):
                    print("Issues at %d!" % i)
                    return out_images, out_boxes, out_classes
                if image == "":
                    image = image_id
                    objects = []
                    classes = []
                    if self._is_more_than_zero(object_bb) and self._is_not_degenerate(object_bb):
                        objects.append(object_bb)
                        classes.append(class_of_object)
                elif image == image_id:
                    if self._is_more_than_zero(object_bb) and self._is_not_degenerate(object_bb):
                        objects.append(object_bb)
                        classes.append(class_of_object)
                elif image != image_id:
                    image_name = self._search_filename_by_imageID(data, image)
                    if isfile(join(images_path, image_name)):
                        img = Image.open(join(images_path, image_name)).convert("RGB")
                        img_width, img_height = img.size
                        if self._is_in_bounds(objects, img_width, img_height):
                            out_images.append(join(images_path, image_name))
                            out_boxes.append(objects)
                            out_classes.append(classes)
                    image = image_id
                    objects = []
                    classes = []
                    if self._is_more_than_zero(object_bb) and self._is_not_degenerate(object_bb):
                        objects.append(object_bb)
                        classes.append(class_of_object)
                    
            image_name = self._search_filename_by_imageID(data, image)
            if isfile(join(images_path, image_name)):
                        img = Image.open(join(images_path, image_name)).convert("RGB")
                        img_width, img_height = img.size
                        if self._is_in_bounds(objects, img_width, img_height):
                            out_images.append(join(images_path, image_name))
                            out_boxes.append(objects)
                            out_classes.append(classes)
        list_of_classes.sort()
        print(list_of_classes)
        print(len(list_of_classes))
        """
        o1 = []
        o2 = []
        o3 = []
        for i in range(50):
            o1.append(out_images[i])
            o2.append(out_boxes[i])
            o3.append(out_classes[i])
        return o1, o2, o3
        """
        return out_images, out_boxes, out_classes
        