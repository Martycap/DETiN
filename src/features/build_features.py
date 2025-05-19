import cv2
import numpy as np
import torch
from scipy.fftpack import fft2, fftshift

def load_image(path, rgb = True):
    img = cv2.imread(path)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img, dtype=torch.float32)
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def srm(img):
    """
    Fridrich et al. 2012
    """

    filters = np.array([
                [[0, 0, 0], [1, -2, 1], [0, 0, 0]],   
                [[0, 1, 0], [0, -2, 0], [0, 1, 0]],      
                [[1, 0, -1], [0, 0, 0], [-1, 0, 1]]
                ])

    filtered = []
    for filter in filters:
        filtered.append(cv2.filter2D(img, -1, filter))

    stacked = np.stack(filtered, axis=0)
    srm_tensor = torch.tensor(stacked, dtype=torch.float32)
    return srm_tensor

def fft(img):
    fft = np.log(1 + np.abs(fftshift(fft2(img))))
    fft_norm = (fft - fft.min()) / (fft.max() - fft.min())
    fft_tensor = torch.tensor(fft_norm, dtype=torch.float32).unsqueeze(0)
    return fft_tensor

img = load_image("data/test/cat.jpg")
gray_img = load_image("data/test/cat.jpg", rgb = False)

srm_tensor = srm(gray_img)
fft_tensor = fft(gray_img)

print("Image Tensor:", img)
print("SRM Tensor:", srm_tensor)
print("FFT Tensor:", fft_tensor)
