import gc

import torch

from baseline_VHR.data_loaders import train_test_split, VHRDataset
from baseline_VHR.torch_utils.engine import train_one_epoch, evaluate
from baseline_VHR.faster_RCNN_baseline import get_fasterRCNN_resnet
import baseline_VHR.torch_utils.utils as utils


def train_model(model, device, dataset, dataset_test, num_epochs=10):
    gc.collect()
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=10, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=10, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    # model = get_object_detection_model(num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(num_epochs):
        # training for one epoch
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
    path = "./VHR_densenet201.pth"
    filepath = "./VHR_statedict_densenet201.pth"
    torch.save(model, path)
    torch.save(model.state_dict(), filepath)


num_classes = 11
params = {'BATCH_SIZE': 32,
          'LR': 0.001,
          'PRECISION': 32,
          'CLASSES': num_classes,
          'SEED': 42,
          'PROJECT': 'Heads',
          'EXPERIMENT': 'heads',
          'MAXEPOCHS': 500,
          'BACKBONE': 'densenet201',
          'ANCHOR_SIZE': ((32, 64, 128, 256, 512),),
          'ASPECT_RATIOS': ((0.5, 1.0, 2.0),),
          'MIN_SIZE': 1024,
          'MAX_SIZE': 1024,
          'IMG_MEAN': [0.485, 0.456, 0.406],
          'IMG_STD': [0.229, 0.224, 0.225],
          'IOU_THRESHOLD': 0.5
          }

model = get_fasterRCNN_resnet(num_classes=params['CLASSES'],
                              backbone_name=params['BACKBONE'],
                              anchor_size=params['ANCHOR_SIZE'],
                              aspect_ratios=params['ASPECT_RATIOS'],
                              min_size=params['MIN_SIZE'],
                              max_size=params['MAX_SIZE'])

dataset, dataset_test, dataset_val = train_test_split(VHRDataset, validation_flag=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_model(model, device, dataset, dataset_test, num_epochs=15)
