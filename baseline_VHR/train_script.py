import gc

import torch

from baseline_VHR.data_loaders import train_test_split, VHRDataset
from baseline_VHR.torch_utils.engine import train_one_epoch, evaluate
from baseline_VHR.faster_RCNN_baseline import get_fasterRCNN_resnet
import baseline_VHR.torch_utils.utils as utils


def train_model(model,
                device,
                dataset,
                dataset_test,
                num_epochs=10):

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
    path = "./VHR_densenet121.pth"
    filepath = "./VHR_statedict_densenet121.pth"
    torch.save(model, path)
    torch.save(model.state_dict(), filepath)
