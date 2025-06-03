import numpy as np
import torch
from PIL import Image
import torch.nn.functional as nnf

def extract_frequency(image_tensor):
    """
    Computes the frequency domain representation of an input RGB image tensor.

    For each channel, the function applies a 2D Fast Fourier Transform (FFT),
    shifts the zero-frequency component to the center, computes the logarithmic
    magnitude spectrum, and normalizes it to the [0, 1] range.
    """
    if isinstance(image_tensor, np.ndarray):
        # da HWC [0,255] → CHW [0,1]
        image_tensor = torch.from_numpy(image_tensor).permute(2, 0, 1).float() / 255.0
        image_tensor = nnf.interpolate(image_tensor.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze(0)

    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("Input must be a NumPy array or a PyTorch tensor")

    if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
        raise ValueError("Expected an RGB image with shape (3, H, W)")
    
    freq_channels = []
    for c in range(image_tensor.shape[0]):
        channel = image_tensor[c]
        fft = torch.fft.fft2(channel)
        fft_shifted = torch.fft.fftshift(fft)
        magnitude = torch.abs(fft_shifted)
        magnitude = torch.log1p(magnitude)
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        freq_channels.append(magnitude)
    freq = torch.stack(freq_channels)
    return freq

def extract_noise(image_tensor):
    """
    Extracts the high-frequency noise residual from an input RGB image tensor.

    The function applies a 3x3 mean filter (box filter) to obtain a blurred version
    of the image, then computes the residual by subtracting the blurred image from
    the original input. This residual captures local high-frequency information,
    often associated with tampering artifacts or compression noise.
    """
    if isinstance(image_tensor, np.ndarray):
        # da HWC [0,255] → CHW [0,1]
        image_tensor = torch.from_numpy(image_tensor).permute(2, 0, 1).float() / 255.0
        image_tensor = nnf.interpolate(image_tensor.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze(0)

    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("Input must be a NumPy array or a PyTorch tensor")

    if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
        raise ValueError("Expected an RGB image with shape (3, H, W)")
    kernel = torch.ones((3,3), device=image_tensor.device) / 9
    kernel = kernel.expand(3,1,3,3)
    img = image_tensor.unsqueeze(0)  
    blurred = nnf.conv2d(img, kernel, padding=1, groups=3)
    noise = img - blurred  
    return noise.squeeze(0)

def center_crop(image, size=256):
    """
    Crops the largest possible centered square from the image,
    then resizes it to (size, size) for inpaint detection model input.
    """
    
    width, height = image.size
    min_dim = min(width, height)

    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim

    cropped = image.crop((left, top, right, bottom))
    resized = cropped.resize((size, size), Image.LANCZOS)
    return resized