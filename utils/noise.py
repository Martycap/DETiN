import torch
import torchvision.transforms.functional as F
import torch.nn.functional as nnf

def extract_noise(image_tensor):
    # image_tensor: [3, H, W] float32 0..1
    # esempio: rumore residuo sottraendo immagine filtrata
    # filtro blur (media)
    kernel = torch.ones((3,3), device=image_tensor.device) / 9
    kernel = kernel.expand(3,1,3,3)
    
    img = image_tensor.unsqueeze(0)  # [1,3,H,W]
    blurred = nnf.conv2d(img, kernel, padding=1, groups=3)
    noise = img - blurred  # residuo
    return noise.squeeze(0)
