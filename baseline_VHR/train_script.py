import gc
import torch
import baseline_VHR.torch_utils.utils as utils

from baseline_VHR.constants.train_constants import BATCH_SIZE_TRAIN, BATCH_SIZE_TEST
from baseline_VHR.torch_utils.engine import train_one_epoch, evaluate
from baseline_VHR.faster_RCNN_baseline import get_fasterRCNN_resnet



def train_model(model,
                device,
                dataset,
                dataset_test,
                num_epochs: int = 2, 
                model_name: str = "1"):

    gc.collect()
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=0,
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
        torch.cuda.empty_cache()
        # training for one epoch
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
    model_name
    path = f"/home/hdd/models/{model_name}.pth"
    filepath = f"/home/hdd/models/{model_name}_1.pth"
    torch.save(model, path)
    torch.save(model.state_dict(), filepath)
