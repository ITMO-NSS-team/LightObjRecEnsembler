
from torch.utils.data import Subset
from torch import randperm, manual_seed
from torch_utils.transforms import ToTensor, RandomHorizontalFlip, Compose


def train_test_split(dataset_class, validation_flag: bool = False):
    """
    Method split dataset's list into two lists 

        :return list - YOLO coordinates
    
    """
    dataset = dataset_class()
    dataset_test = dataset_class()
    dataset_validation = dataset_class()
    # split the dataset in train and test set
    manual_seed(1)
    indices = randperm(len(dataset)).tolist()

    # train test split
    test_split = 0.2
    valida_split = 0.2
    tsize = int(len(dataset) * test_split)
    vsize = int(len(dataset) * test_split * valida_split)
    dataset = Subset(dataset, indices[:-tsize])
    dataset_test = Subset(dataset_test, indices[-tsize:-vsize])
    if validation_flag:
        dataset_validation = Subset(dataset_validation, indices[-vsize:])
    else:
        dataset_validation = None

    return dataset, dataset_test, dataset_validation

def _get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)
