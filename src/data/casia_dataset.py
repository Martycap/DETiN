import torch
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
from utils.frequency import extract_frequency
from utils.noise import extract_noise
import matplotlib.pyplot as plt


class CASIATransformerDataset(Dataset):
    """
    Custom PyTorch dataset for the CASIA dataset.

    Each item consists of:
    - A tampered image
    - Its corresponding binary manipulation mask
    - Additional feature maps: noise and frequency

    The tampered image is preprocessed (resized and normalized), and
    concatenated with its noise and frequency maps to form a 9-channel tensor.
    """
    def __init__(self, triplets, transform=None):
        self.triplets = triplets
        self.transform = transform if transform else transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        """
        Returns:
            x (Tensor): A 9-channel tensor containing RGB, noise, and frequency maps.
            mask (Tensor): A 1-channel ground truth binary mask of shape [1, 224, 224].
        """
        original_path, tampered_path, mask_path = self.triplets[idx]
        
        image = cv2.imread(tampered_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (224,224))
        mask = torch.tensor(mask / 255., dtype=torch.float32)  
        
        image = self.transform(image)  
        
        
        noise = extract_noise(image)       
        freq = extract_frequency(image)    
        
        
        x = torch.cat([image, noise, freq], dim=0)
        
        return x, mask.unsqueeze(0)  
    
    
    
def plot_image_noise_freq(image, noise, freq):
    """
    Plots the first channel of the original image, noise map, and frequency map.

    Args:
        image (Tensor): Original image tensor of shape [3, H, W]
        noise (Tensor): Noise map tensor of shape [3, H, W]
        freq (Tensor): Frequency map tensor of shape [3, H, W]
    """
    img_np = image[0].cpu().numpy()
    noise_np = noise[0].cpu().numpy()
    freq_np = freq[0].cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15,5))
    axs[0].imshow(img_np, cmap='gray')
    axs[0].set_title('Immagine (canale 0)')
    axs[0].axis('off')

    axs[1].imshow(noise_np, cmap='gray')
    axs[1].set_title('Rumore (canale 0)')
    axs[1].axis('off')

    axs[2].imshow(freq_np, cmap='gray')
    axs[2].set_title('Frequenza (canale 0)')
    axs[2].axis('off')

    plt.show()
