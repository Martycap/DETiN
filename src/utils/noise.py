import torch
import torch.nn.functional as nnf

def extract_noise(image_tensor):
    """
    Extracts the high-frequency noise residual from an input RGB image tensor.

    The function applies a 3x3 mean filter (box filter) to obtain a blurred version
    of the image, then computes the residual by subtracting the blurred image from
    the original input. This residual captures local high-frequency information,
    often associated with tampering artifacts or compression noise.
    """

    kernel = torch.ones((3,3), device=image_tensor.device) / 9
    kernel = kernel.expand(3,1,3,3)
    img = image_tensor.unsqueeze(0)  
    blurred = nnf.conv2d(img, kernel, padding=1, groups=3)
    noise = img - blurred  
    return noise.squeeze(0)
