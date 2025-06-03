import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import models
import tqdm, os, json

class DETIN(nn.Module):
    def __init__(self, input_channels=(3, 3, 3), num_classes=1, pretrained=False):
        super().__init__()
        self.rgb_branch = self._make_branch(input_channels[0], pretrained)
        self.freq_branch = self._make_branch(input_channels[1], pretrained)
        self.noise_branch = self._make_branch(input_channels[2], pretrained)

        fusion_channels = 3 * 2048
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_channels, 2048, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def _make_branch(self, in_channels, pretrained):
        model = models.resnet50(pretrained=pretrained)
        model.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        return nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        )

    def forward(self, x):
        rgb = x[:, :3, :, :]
        freq = x[:, 3:6, :, :]
        noise = x[:, 6:, :, :]

        rgb_feat = self.rgb_branch(rgb)
        freq_feat = self.freq_branch(freq)
        noise_feat = self.noise_branch(noise)

        x = torch.cat([rgb_feat, freq_feat, noise_feat], dim=1)
        x = self.fusion(x)
        x = self.classifier(x)
        x = nn.functional.interpolate(x, scale_factor=32, mode="bilinear", align_corners=False)
        return {'out': x}
    
def prepare_model(input_channels=9, num_classes=1, pretrained=False):
    return DETIN(input_channels=(3, 3, 3), num_classes=num_classes, pretrained=pretrained)

# import torch.nn as nn
# import torchvision.models.segmentation as segmentation

# def prepare_model(input_channels=9, num_classes=1, pretrained=False):
#     model = segmentation.deeplabv3_resnet50(pretrained=pretrained, num_classes=num_classes)
#     old_conv = model.backbone.conv1
#     model.backbone.conv1 = nn.Conv2d(
#         in_channels=input_channels,
#         out_channels=old_conv.out_channels,
#         kernel_size=old_conv.kernel_size,
#         stride=old_conv.stride,
#         padding=old_conv.padding,
#         bias=old_conv.bias is not None
#     )
#     return model
