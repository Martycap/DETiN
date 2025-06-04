import torch, torch.nn as nn
from torchvision import models

class DETiN(nn.Module):
    def __init__(self, num_classes, pretrained):
        super().__init__()
        self.rgb_branch = self._make_branch(pretrained)
        self.freq_branch = self._make_branch(pretrained)
        self.noise_branch = self._make_branch(pretrained)

        fusion_channels = 3 * 2048
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_channels, 2048, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3)
        )

        self.prediction = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def _make_branch(self, pretrained):
        model = models.resnet50(pretrained=pretrained)
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
        x = self.prediction(x)
        x = nn.functional.interpolate(x, scale_factor=32, mode="bilinear", align_corners=False)
        return {'out': x}
    
    
# ------------------------------------------------------------------------------
from torchvision.models.segmentation import deeplabv3_resnet50

# First model tested.
def prepare_model(input_channels=9, num_classes=1, pretrained=False):
    model = deeplabv3_resnet50(pretrained=pretrained, num_classes=num_classes)
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