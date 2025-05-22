import torch
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
from utils.frequency import extract_frequency
from utils.noise import extract_noise
import matplotlib.pyplot as plt
import cv2

class CASIATransformerDataset(Dataset):
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
        original_path, tampered_path, mask_path = self.triplets[idx]
        
        # Carica immagine modificata
        image = cv2.imread(tampered_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Carica maschera (bianco/nero)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (224,224))
        mask = torch.tensor(mask / 255., dtype=torch.float32)  # [H, W], valori 0 o 1
        
        # Applica trasformazioni immagine
        image = self.transform(image)  # [3,H,W]
        
        # Estrai rumore e frequenza
        noise = extract_noise(image)       
        freq = extract_frequency(image)    
        
        # Concatenazione canali: immagine + rumore + frequenza -> [9,H,W]
        x = torch.cat([image, noise, freq], dim=0)
        
        return x, mask.unsqueeze(0)  
    
    
    
def plot_image_noise_freq(image, noise, freq):
    # image, noise, freq: [3, H, W], tensori torch

    # Converti in numpy e prendi solo il primo canale per visualizzare (per semplicità)
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
