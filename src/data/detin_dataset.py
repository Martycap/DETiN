import torch, cv2, os, sys
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import random

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))
from features.build_features import extract_frequency
from features.build_features import extract_noise


class DETiNTransformerDataset(Dataset):
    """
    Custom PyTorch dataset for the DETIN model.

    Each item consists of:
    - A tampered image
    - Its corresponding binary manipulation mask
    - Additional feature maps: noise and frequency

    The tampered image is preprocessed (resized and normalized), and
    concatenated with its noise and frequency maps to form a 9-channel tensor.
    """
    def __init__(self, pairs, transform=None):
        random.shuffle(pairs)
        self.pairs = pairs
        self.transform = transform if transform else transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        self.resize = transforms.Resize((224,224))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Returns:
            x (Tensor): A 9-channel tensor containing RGB, noise, and frequency maps.
            mask (Tensor): A 1-channel ground truth binary mask of shape [1, 224, 224].
        """
        tampered_path, mask_path = self.pairs[idx]
        
        image = cv2.imread(tampered_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (224,224))
        mask = torch.tensor(mask / 255., dtype=torch.float32)  

        image = self.transform(image)
        
        noise = extract_noise(image)       
        freq = extract_frequency(image)
        
        image = self.resize(image)
        noise = self.resize(noise)
        freq = self.resize(freq)
        
        x = torch.cat([image, noise, freq], dim=0)
        filename = os.path.splitext(os.path.basename(tampered_path))[0]
        
        return x, mask.unsqueeze(0), filename
    

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

    _, axs = plt.subplots(1, 3, figsize=(15,5))
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
