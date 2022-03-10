import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from os.path import join
from os.path import splitext
from torchvision import transforms
from baseline_VHR.data_loaders.data_constants import YOLO_PATH_TO_IMAGES, YOLO_PATH_TO_ANNOTATIONS


class YOLODataset(torch.utils.data.Dataset):
    """
    Class-loader for YOLO-formated datasets
    """

    def __init__(self):
        self.imgs, self.boxes, self.classes = self._read_yolov4(YOLO_PATH_TO_IMAGES, YOLO_PATH_TO_ANNOTATIONS)

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


    def _from_yolo_to_rec(self, box: list, image_width: int, image_height: int) -> list:
        """
        This method turns annotations in YOLO format into rectangle coordinates
        """
        box_w = int(box[2] * image_width)
        box_h = int(box[3] * image_height)
        x_mid = int(box[0] * image_width + 1)
        y_mid = int(box[1] * image_height + 1)
        x_min = int(x_mid - box_w / 2) + 1
        x_max = int(x_mid + box_w / 2) - 1
        y_min = int(y_mid - box_h / 2) + 1
        y_max = int(y_mid + box_h / 2) - 1
        return [x_min, y_min, x_max, y_max]


    def _read_yolov4(self, images_path: str, annotations_path: str) -> list:
        """
        This method reads images and annotations in YOLOv4 format and returns lists of images, boxes and classes

        :param images_path - path to images folder
        :param annotations_path - path to annotations folder

        :return out_images - list of images
        :return out_boxes - list of list of boxes
        :return out_classes - list of list of classes
        """
        out_images = []
        out_boxes = []
        out_classes = []
        img_ext = [".jpg", ".png", ".JPG", ".PNG"]
        txt_ext = [".txt", ".TXT"]
        image_list = next(os.walk(images_path))[2]
        image_list.sort()
        txt_list = next(os.walk(annotations_path))[2]
        txt_list.sort()
        if len(image_list) == len(txt_list):
            pair_count = len(txt_list)
            is_all_files_paired = True
            for i in tqdm(range(pair_count), colour="red"):
                img_pathname, img_extension = splitext(image_list[i])
                txt_pathname, txt_extension = splitext(txt_list[i])
                if not (img_pathname == txt_pathname and img_extension in img_ext and txt_extension in txt_ext):
                    print(f"ERROR! Pair {i}: img {image_list[i]} - txt {txt_list[i]}")
                    is_all_files_paired = False
            if is_all_files_paired:
                print(f"All images have a respective pair of text files!")
                print(f"Pair count is {pair_count}")
                for i in tqdm(range(pair_count), colour="blue"):
                    out_images.append(join(images_path, image_list[i]))

                    objects_classes = []
                    objects = []
                    rec_objects = []
                    img = Image.open(join(images_path, image_list[i]))
                    img_width, img_height = img.size
                    with open(join(annotations_path, txt_list[i])) as f:
                        lines = f.readlines()
                        for line in lines:
                            line1 = line.strip(' \n').split(' ')
                            float_line = list(np.float_(line1))
                            objects_classes.append(int(float_line[0]))
                            float_line.pop(0)
                            objects.append(float_line)
                        for i in range(len(objects)):
                            rec_objects.append(self._from_yolo_to_rec(objects[i], img_width, img_height))
                        out_boxes.append(rec_objects)
                        out_classes.append(objects_classes)
            else:
                print(f"ERROR! Not all images have a respective pair of text files!")
                return [], []
        elif len(image_list) > len(txt_list):
            print(f"ERROR! Not all images have a respective pair of text files!")
            print(f"There is {len(image_list) - len(txt_list)} images without pairs")
            return [], []
        else:
            print(f"ERROR! Not all images have a respective pair of text files!")
            print(f"There is {len(txt_list) - len(image_list)} txt files without pairs")
            return [], []
        return out_images, out_boxes, out_classes

