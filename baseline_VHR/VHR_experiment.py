import gc
import itertools
import os

import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms

from baseline_VHR.train_script import train_model
from model_params import *
import baseline_VHR.torch_utils.transforms as T
from baseline_VHR.data_loaders import train_test_split, VHRDataset
from baseline_VHR.visualization import plot_img_bbox
from baseline_VHR.faster_RCNN_baseline import get_fasterRCNN_resnet
from baseline_VHR.utils.ensemble import Rectangle
from baseline_VHR.filtering_ensembel import filtering_ensemble
from baseline_VHR.validation_weights import validation_weights, calculate_coco_metrics
from baseline_VHR.utils.utils import visualise_model_prediction, visualise_model_prediction_nms


class PerformExperiment:
    def __init__(self,
                 model_list: list,
                 num_classes: int = 11,
                 params: dict = None):
        self.model_list = model_list
        self.num_classes = num_classes
        self.params = params
        self.columns = [
            "AP",
            "AP50",
            "AP75",
            "APsmall",
            "APmedium",
            "APlarge",
            "AR1",
            "AR10",
            "AR100",
            "ARsmall",
            "ARmedium",
            "ARlarge"
        ]
        return

    def get_transform(self, train):
        transforms = []
        transforms.append(T.ToTensor())
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)

    def torch_to_pil(self,
                     img):
        return transforms.ToPILImage()(img).convert('RGB')

    def _rectangle_intersect(self,
                             first_box: list,
                             second_box: list):
        a = Rectangle(*first_box)
        b = Rectangle(*second_box)
        area = a & b
        if area is None:
            return area, None
        else:
            intersect_area = area.calculate_area()
            # composite_bbox = a - area
            composite_bbox = a.difference(area)
            ratio_1, ratio_2 = intersect_area / a.calculate_area(), intersect_area / b.calculate_area()
            return (ratio_1, ratio_2), composite_bbox

    def ensemble_OD_predictions(self,
                                bboxes: list,
                                labels: list,
                                scores: list,
                                weights: list,
                                area_threshold: float = 0.75):
        best_model_ind = np.argmax(weights)
        weak_model_ind = 1 if best_model_ind == 0 else 0

        bboxes_merged = []
        labels_merged = []
        scores_merged = []

        chosen_box = None
        chosen_label = None
        chosen_score = None
        check_flag = True

        for index_1, first_box in enumerate(bboxes[weak_model_ind]):
            for index_2, second_box in enumerate(bboxes[best_model_ind]):
                ratio, intersect_coord = self._rectangle_intersect(first_box, second_box)
                if ratio is None:
                    check_flag = True
                elif sum(ratio) > area_threshold:
                    CW1 = weights[best_model_ind] * scores[weak_model_ind][index_1]
                    CW2 = weights[weak_model_ind] * scores[best_model_ind][index_2]
                    if labels[weak_model_ind][index_1] == labels[best_model_ind][index_2]:
                        chosen_box = second_box
                        chosen_label = labels[best_model_ind][index_2]
                        chosen_score = scores[best_model_ind][index_2]
                    else:
                        best_CW_ind = np.argmax([CW1, CW2])
                        chosen_box = first_box if best_CW_ind == 0 else second_box
                        chosen_ind = index_1 if best_CW_ind == 0 else index_2
                        chosen_label = labels[best_CW_ind][chosen_ind]
                        chosen_score = scores[best_CW_ind][chosen_ind]

                    check_flag = False
                    break

            if check_flag and chosen_box is not None:
                bboxes_merged.append(chosen_box)
                labels_merged.append(chosen_label)
                scores_merged.append(chosen_score)

        compose_bbox = list(itertools.chain(bboxes[best_model_ind], bboxes_merged)),
        compose_labels = list(itertools.chain(labels[best_model_ind], labels_merged))
        compose_scores = list(itertools.chain(scores[best_model_ind], scores_merged))

        return {'boxes': np.array(compose_bbox[0]), 'labels': np.array(compose_labels),
                'scores': np.array(compose_scores)}

    def get_model(self, model_name):
        model_params = self.params[model_name]
        model = get_fasterRCNN_resnet(**model_params)
        return model

    def _get_dataframes_for_exp_result(self, model_list):
        df_dict = dict()
        df_list = []
        for model in model_list:
            df_list.append((model, pd.DataFrame(columns=self.columns)))
        df_dict.update(df_list)
        return df_dict

    def load_model_weights(self, filepath_list: list):
        self.loaded_model_list = []
        for model_name, filepath in zip(self.model_list, filepath_list):
            model = self.get_model(model_name)
            model.load_state_dict(torch.load(filepath))
            model.eval()
            self.loaded_model_list.append(model)

    def fit(self):
        for model_name in self.model_list:
            model = self.get_model(model_name)
            train_model(model, device, dataset, dataset_test, num_epochs=15)

    def predict(self, dataset_test):
        image_ids = []
        for i in range(len(dataset_test)):
            result_current_image = pd.DataFrame()
            img, target = dataset_test[i]
            image_id = str(target['image_id'].tolist()[0] + 1)
            image_ids.append(image_id)
            plot_img_bbox(img, target, title='IMAGE', save=save,
                          image_id=image_id, show=show, path=path_prediction)

            with torch.no_grad():
                all_prediction = []
                all_metrics = []
                tmp_model_list = self.model_list.copy()
                for model_name, loaded_model in zip(self.model_list, self.loaded_model_list):
                    prediction = loaded_model([img.to(device)])[0]
                    ### VISUALISE MODELS PREDICTIONS
                    metrics = visualise_model_prediction(prediction, target, img, image_id, save, show,
                                                         path_prediction, model_name)

                    ### VISUALISE MODELS PREDICTIONS AFTER IOU_THERESHOLD
                    nms_prediction, metrics_nms = visualise_model_prediction_nms(prediction, target,
                                                                                 img, image_id, save,
                                                                                 show, path_prediction,
                                                                                 model_name.format('_NMS'))

                    all_prediction.append(nms_prediction)
                    all_metrics.append(metrics)
                    tmp_model_list.append(model_name.format('_NMS'))
                    all_metrics.append(metrics_nms)

                    ### VISUALISE MODELS PREDICTIONS AFTER ENSEMBLING
                all_prediction = filtering_ensemble(all_prediction, val_weights, image_id)
                all_val_weights = val_weights.copy()

                while len(all_prediction) != 1:
                    inter_val_weights = []
                    nms_pred = all_prediction.pop()
                    nms_pred_2 = all_prediction.pop()
                    inter_val_weights.append(all_val_weights.pop())
                    inter_val_weights.append(all_val_weights.pop())
                    compose_bbox = self.ensemble_OD_predictions([nms_pred['boxes'], nms_pred_2['boxes']],
                                                                [nms_pred['labels'], nms_pred_2['labels']],
                                                                [nms_pred['scores'], nms_pred_2['scores']],
                                                                weights=inter_val_weights)
                    all_prediction.append(compose_bbox)
                    all_val_weights.append(max(inter_val_weights))

                compose_bbox = all_prediction[0]

                plot_img_bbox(img, compose_bbox, title='ENSEMBLE', save=True,
                              image_id=image_id, show=show, path=path_prediction)
                metrics_ensemble_nms = calculate_coco_metrics(target, compose_bbox)
                all_metrics.append(metrics_ensemble_nms)
                tmp_model_list.append('ensemble')
                ### SAVE METRICS FOR CURRENT IMAGE

                df_dict = self._get_dataframes_for_exp_result(tmp_model_list)
                result_df_list = []

                for model_name, metrics in zip(tmp_model_list, all_metrics):
                    tmp_df = df_dict[model_name].append(metrics, ignore_index=True)
                    tmp_df['image_id'] = image_ids
                    result_df_list.append(tmp_df)

                ### SAVE CSV FILES
                for res, name in zip(result_df_list, tmp_model_list):
                    res.to_csv(os.path.join(path_prediction, f'{name}.csv'))

                for name, metric in zip(tmp_model_list, all_metrics):
                    result_current_image[name] = list(metric.values())
                result_current_image.index = self.columns
                result_current_image.to_csv(os.path.join(path_prediction, image_id, f'{image_id}.csv'))

        return result_df_list


if __name__ == '__main__':
    save = False
    show = False

    num_classes = 11
    model_list = ['fasterrcnn_resnet50_fpn',
                  'resnet18',
                  'mobilenet_v3_large',
                  'densenet121']

    filepath_list = ["../local/VHR_statedict_fasterrcnn_resnet50_fpn.pth",
                     "../local/VHR_statedict_resnet18.pth",
                     "../local/VHR_statedict_mobilenet_v3_large.pth",
                     "../local/VHR_statedict_densenet121.pth"]
    val_weights = [0.5279900411868104, 0.04605057596583206, 0.11095235215754355, 0.31500703068981406]

    dataset, dataset_test, dataset_val = train_test_split(VHRDataset, validation_flag=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    path = os.path.dirname(os.path.abspath(__file__))
    path_prediction = os.path.join(path, 'NWPU VHR-10 dataset', 'last_prediction_4_models')

    # TODO apply_nms after ensemble

    gc.collect()
