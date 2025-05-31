import torch
import torch.nn as nn
import torch.nn.functional as F

class ChangeDetector(nn.Module):
    """
    A convolutional neural network for binary change detection in paired images.

    The model takes as input two concatenated RGB images (resulting in 6 channels) and outputs a 
    binary segmentation mask highlighting the regions where changes have occurred.

    Architecture:
        - Encoder: 3 convolutional layers with ReLU activations and max pooling.
        - Decoder: 3 transposed convolutional layers with bilinear upsampling and ReLU activations.
        - Output: A single-channel mask passed through a sigmoid activation to produce values in [0, 1].

    Forward Input:
        x (torch.Tensor): A tensor of shape (batch_size, 6, H, W), where the 6 channels correspond 
                          to two RGB images stacked along the channel dimension.

    Forward Output:
        torch.Tensor: A tensor of shape (batch_size, 1, H, W) representing the predicted change mask.
    """
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(6, 16, 3, padding=1)
        self.enc2 = nn.Conv2d(16, 32, 3, padding=1)
        self.enc3 = nn.Conv2d(32, 64, 3, padding=1)

        # Decoder
        self.dec3 = nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.dec2 = nn.ConvTranspose2d(32, 16, 3, padding=1)
        self.dec1 = nn.ConvTranspose2d(16, 1, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.enc2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.enc3(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.relu(self.dec3(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.relu(self.dec2(x))
        x = torch.sigmoid(self.dec1(x))  

        return x