import gc
import itertools
import os
import sys
sys.path.append ("/home/nikita/Desktop/NAS-object-recognition")
#sys.path.append ("/home/NAS-object-recognition")
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
import baseline_VHR.torch_utils.transforms as T
from tqdm import tqdm


from baseline_VHR.train_script import train_model
from baseline_VHR.faster_RCNN_baseline import get_fasterRCNN_resnet

# Model's params. IMPORTANT!
from baseline_VHR.constants.model_params import *
# Datasets' loaders
from baseline_VHR.data_loaders.xView_dataloader import xViewDataset
from baseline_VHR.data_loaders.METU_dataloader import METUDataset
from baseline_VHR.data_loaders.YOLO_dataloader import YOLODataset
#Datasets' things
from baseline_VHR.data_loaders.data_constants import METU_TEST, METU_TRAIN, METU_VAL
from baseline_VHR.data_utils.data_split import train_test_split

from baseline_VHR.constants.train_constants import EPOCH_NUMBER

from baseline_VHR.visualization import plot_img_bbox
from baseline_VHR.utils.utils import torch_to_pil
from baseline_VHR.utils.bounding_boxes_utils import rectangle_intersect
from baseline_VHR.filtering_ensembel import filtering_ensemble
from baseline_VHR.validation_weights import validation_weights, calculate_coco_metrics
from baseline_VHR.utils.utils import visualise_model_prediction, visualise_model_prediction_nms, get_model_metrics, get_model_metrics_nms


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class PerformExperiment:
    """
    Main class of projects, contains models and functions for loading, fitting, testing and ensembling.
    
    """
    def __init__(self,
                 model_list: list,
                 num_classes: int = 62,
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

    def get_transform(self, train: bool):
        """
        Method returns transform for data.

        :param train - bool

        :return transform objects
        """
        transforms = []
        transforms.append(T.ToTensor())
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)

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
                ratio, intersect_coord = rectangle_intersect(first_box, second_box)
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

    def get_model(self, model_name: str):
        """
        Method-creator empty model for load_model_weights

        :param model_name - name of model
        :return model - empty model
        """
        model_params = self.params[model_name]
        model = get_fasterRCNN_resnet(**model_params)
        return model
    
    def load_model_weights(self, list_of_models_name: list):
        """
        Method-loader for models. Need some work still.

        :param list_of_models_name - list of models' names 
        """
        self.loaded_model_list = []
        for model_name in list_of_models_name:
            model = self.get_model(model_name)
            #model.load_state_dict(torch.load(f"/home/hdd/models/{model_name}_1.pth"))
            model.load_state_dict(torch.load(f"/media/nikita/HDD/models/{model_name}_1.pth"))
            model.eval()
            self.loaded_model_list.append(model)

    def _get_dataframes_for_exp_result(self, model_list):

        df_dict = dict()
        df_list = []
        for model in model_list:
            df_list.append((model, pd.DataFrame(columns=self.columns)))
        df_dict.update(df_list)
        return df_dict

    def fit(self, train_dataset, dataset_val):
        """
        Method-starter for fitting of models. One by one it creates and fits all the models in the self.model_list. 
        After trainings models will be save in files.

        :param train_dataset - dataset for training
        """
        for model_name in self.model_list:
            model = self.get_model(model_name)
            train_model(model, device, train_dataset, dataset_val, num_epochs=EPOCH_NUMBER, model_name=model_name)

    def predict(self, dataset_test):
        """
        Method for test prediction
        
        """
        image_ids = []
        for i in tqdm(range(len(dataset_test))):
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
                    loaded_model.to(device)
                    prediction = loaded_model([img.to(device)])[0]
                    ### VISUALISE MODELS PREDICTIONS
                    metrics = visualise_model_prediction(prediction, target, img, image_id, save, show,
                                                         path_prediction, model_name)

                    ### VISUALISE MODELS PREDICTIONS AFTER IOU_THERESHOLD
                    nms_prediction, metrics_nms = visualise_model_prediction_nms(prediction, target,
                                                                                 img, image_id, save,
                                                                                 show, path_prediction,
                                                                                 f"{model_name}_NMS")

                    all_prediction.append(nms_prediction)
                    all_metrics.append(metrics)
                    tmp_model_list.append(f"{model_name}_NMS")
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
                    #tmp_df['image_id'] = image_ids
                    #tmp_df['image_id'] = tmp_df.get('image_id', []) + image_ids
                    result_df_list.append(tmp_df)

                ### SAVE CSV FILES
                for res, name in zip(result_df_list, tmp_model_list):
                    res.to_csv(os.path.join(path_prediction, f'{name}.csv'))

                for name, metric in zip(tmp_model_list, all_metrics):
                    result_current_image[name] = list(metric.values())
                result_current_image.index = self.columns
                result_current_image.to_csv(os.path.join(path_prediction, image_id, f'{image_id}.csv'))

        return result_df_list


    def predict_vithout_visualisation(self, dataset_test):
        """
        Method for test prediction
        
        """
        image_ids = []
        out_list = []

        for i in tqdm(range(len(dataset_test))):
            result_current_image = pd.DataFrame()
            img, target = dataset_test[i]
            image_id = str(target['image_id'].tolist()[0] + 1)
            image_ids.append(image_id)
            with torch.no_grad():
                all_prediction = []
                all_metrics = []
                tmp_model_list = self.model_list.copy()
                for model_name, loaded_model in zip(self.model_list, self.loaded_model_list):
                    loaded_model.to(device)
                    prediction = loaded_model([img.to(device)])[0]
                    metrics = get_model_metrics(prediction, target)

                    nms_prediction, metrics_nms = get_model_metrics_nms(prediction, target)

                    all_prediction.append(nms_prediction)
                    all_metrics.append(metrics)
                    tmp_model_list.append(f"{model_name}_NMS")
                    all_metrics.append(metrics_nms)

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
                out_list.append(compose_bbox)
                plot_img_bbox(img, compose_bbox, title='ENSEMBLE', save=True,
                              image_id=image_id, show=False, path=path_prediction)
                metrics_ensemble_nms = calculate_coco_metrics(target, compose_bbox)
                all_metrics.append(metrics_ensemble_nms)
                tmp_model_list.append('ensemble')
                ### SAVE METRICS FOR CURRENT IMAGE

                df_dict = self._get_dataframes_for_exp_result(tmp_model_list)
                result_df_list = []

                for model_name, metrics in zip(tmp_model_list, all_metrics):
                    tmp_df = df_dict[model_name].append(metrics, ignore_index=True)
                    #tmp_df['image_id'] = image_ids
                    #tmp_df['image_id'] = tmp_df.get('image_id', []) + image_ids
                    result_df_list.append(tmp_df)

                ### SAVE CSV FILES
                for res, name in zip(result_df_list, tmp_model_list):
                    res.to_csv(os.path.join(path_prediction, f'{name}.csv'))

                for name, metric in zip(tmp_model_list, all_metrics):
                    result_current_image[name] = list(metric.values())
                result_current_image.index = self.columns
                #os.mkdir(os.path.join(path_prediction, f"{image_id}"))
                result_current_image.to_csv(os.path.join(path_prediction, image_id, f'{image_id}.csv'))

        return out_list


if __name__ == '__main__':
    save = False
    show = True

    num_classes = NUM_CLASSES
    model_list = ['fasterrcnn_resnet50_fpn',
                  'resnet18',
                  'mobilenet_v3_large',
                  'densenet121']

    filepath_list = ["../local/VHR_statedict_fasterrcnn_resnet50_fpn.pth",
                     "../local/VHR_statedict_resnet18.pth",
                     "../local/VHR_statedict_mobilenet_v3_large.pth",
                     "../local/VHR_statedict_densenet121.pth"]
    val_weights = [0.5279900411868104, 0.04605057596583206, 0.11095235215754355, 0.31500703068981406]

    experimenter = PerformExperiment(model_list=model_list,
                                     num_classes=num_classes,
                                     params=xView_model_dict)

    dataset = METUDataset(val_flag=METU_TRAIN)
    dataset_val = METUDataset(val_flag=METU_VAL)
    #dataset = xViewDataset()

    #dataset, dataset_test, dataset_val = train_test_split(xViewDataset, validation_flag=True)
    #experimenter.fit(dataset, dataset_val)
    experimenter.load_model_weights(model_list)

    path = os.path.dirname(os.path.abspath(__file__))
    path_prediction = os.path.join(path, 'NWPU VHR-10 dataset', 'last_prediction_4_models')
    #experimenter.predict(dataset_val)
    experimenter.predict_vithout_visualisation(dataset_val)
    gc.collect()
