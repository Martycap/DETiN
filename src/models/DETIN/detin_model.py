import torch.nn as nn
import torchvision.models.segmentation as segmentation

def prepare_model(input_channels=9, num_classes=1, pretrained=False):
    model = segmentation.deeplabv3_resnet50(pretrained=pretrained, num_classes=num_classes)
    old_conv = model.backbone.conv1
    model.backbone.conv1 = nn.Conv2d(
        in_channels=input_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )
    return model
