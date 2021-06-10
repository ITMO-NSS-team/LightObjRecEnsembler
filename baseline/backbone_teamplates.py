import torchvision.models as models
from torchvision.models.detection import faster_rcnn, mask_rcnn, retinanet, keypoint_rcnn
import torch
from torch import nn
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork


def get_resnet_backbone(backbone_name: str):
    """
    Returns a resnet backbone pretrained on ImageNet.
    Removes the average-pooling layer and the linear layer at the end.
    """
    if backbone_name == 'resnet18':
        pretrained_model = models.resnet18(pretrained=True, progress=False)
        out_channels = 512
    elif backbone_name == 'resnet34':
        pretrained_model = models.resnet34(pretrained=True, progress=False)
        out_channels = 512
    elif backbone_name == 'resnet50':
        pretrained_model = models.resnet50(pretrained=True, progress=False)
        out_channels = 2048
    elif backbone_name == 'resnet101':
        pretrained_model = models.resnet101(pretrained=True, progress=False)
        out_channels = 2048
    elif backbone_name == 'resnet152':
        pretrained_model = models.resnet152(pretrained=True, progress=False)
        out_channels = 2048

    backbone = torch.nn.Sequential(*list(pretrained_model.children())[:-2])
    backbone.out_channels = out_channels

    return backbone


def get_densenet_backbone(backbone_name: str):
    """
    Returns a resnet backbone pretrained on ImageNet.
    Removes the linear layer at the end.
    """
    if backbone_name == 'densenet121':
        pretrained_model = models.densenet121(pretrained=True, progress=False)
        out_channels = 1024
    elif backbone_name == 'densenet161':
        pretrained_model = models.densenet161(pretrained=True, progress=False)
        out_channels = 2208
    elif backbone_name == 'densenet169':
        pretrained_model = models.densenet169(pretrained=True, progress=False)
        out_channels = 1664
    elif backbone_name == 'densenet201':
        pretrained_model = models.densenet201(pretrained=True, progress=False)
        out_channels = 1920

    backbone = torch.nn.Sequential(*list(pretrained_model.children())[:-1])
    backbone.out_channels = out_channels

    return backbone


def get_resnet_rcnn_backbone(backbone_name: str):
    """
    Returns a resnet backbone pretrained on ImageNet.
    Removes the average-pooling layer and the linear layer at the end.
    """

    if backbone_name == 'fasterrcnn_resnet50_fpn':
        pretrained_model = faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=True, progress=False)
        out_channels = 512
    elif backbone_name == 'retinanet_resnet50_fpn':
        pretrained_model = retinanet.retinanet_resnet50_fpn(pretrained=True, progress=False)
        out_channels = 512
    elif backbone_name == 'keypointrcnn_resnet50_fpn':
        pretrained_model = keypoint_rcnn.keypointrcnn_resnet50_fpn(pretrained=True, progress=False)
        out_channels = 2048
    elif backbone_name == 'maskrcnn_resnet50_fpn':
        pretrained_model = mask_rcnn.maskrcnn_resnet50_fpn(pretrained=True, progress=False)

    backbone = torch.nn.Sequential(*list(pretrained_model.children())[:-2])
    backbone.out_channels = out_channels

    return backbone


def get_resnet_fpn_backbone(backbone_name: str, pretrained: bool = True, trainable_layers: int = 5):
    """
    Returns a resnet backbone with fpn pretrained on ImageNet.
    """
    backbone = resnet_fpn_backbone(backbone_name=backbone_name,
                                   pretrained=pretrained,
                                   trainable_layers=trainable_layers)

    backbone.out_channels = 256
    return backbone


def resnet_fpn_backbone(backbone_name: str,
                        pretrained: bool,
                        norm_layer=misc_nn_ops.FrozenBatchNorm2d,
                        trainable_layers: int = 3,
                        returned_layers=None,
                        extra_blocks=None
                        ):
    # Slight adaptation from the original pytorch vision package
    # Changes: Removed extra_blocks parameter - This parameter invokes LastLevelMaxPool(), which I don't need
    """
    Constructs a specified ResNet backbone with FPN on top. Freezes the specified number of layers in the backbone.

    Arguments:
        backbone_name (string): resnet architecture. Possible values are 'ResNet', 'resnet18', 'resnet34', 'resnet50',
             'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
        norm_layer (torchvision.ops): it is recommended to use the default value. For details visit:
            (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)
        pretrained (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained,
        norm_layer=norm_layer)

    # select layers that wont be frozen
    assert trainable_layers <= 5 and trainable_layers >= 0
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    # freeze layers only if pretrained backbone is used
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)


class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.
    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(self, backbone, return_layers, in_channels_list, out_channels, extra_blocks=None):
        super(BackboneWithFPN, self).__init__()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x
