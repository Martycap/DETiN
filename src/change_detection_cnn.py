#Da capire se sarà utile
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
from src.utils.create_triplets import create_triplets_from_tampered
import cv2
import pickle
from torchvision import transforms


class CASIATransformerDataset(Dataset):
    def __init__(self, triplets, transform=None):
        self.triplets = triplets
        self.transform = transform if transform else transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        original_path, tampered_path, mask_path = self.triplets[idx]
        print(f"Caricamento: {original_path}, {tampered_path}, {mask_path}")
        
        # Caricamento e conversione immagini
        original = cv2.imread(original_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        tampered = cv2.imread(tampered_path)
        tampered = cv2.cvtColor(tampered, cv2.COLOR_BGR2RGB)

        # Applica le trasformazioni (inclusa normalizzazione) a entrambe
        original = self.transform(original)  # [3, 224, 224]
        tampered = self.transform(tampered)  # [3, 224, 224]

        # Concatenazione lungo i canali -> [6, 224, 224]
        image_pair = torch.cat((original, tampered), dim=0)

        # Caricamento e normalizzazione maschera
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (224, 224))
        mask = torch.tensor(mask / 255., dtype=torch.float32).unsqueeze(0)  # [1, 224, 224]

        return image_pair, mask



class ImagePairDataset(Dataset):
    def __init__(self, original_dir, modified_dirs, transform=None):
        self.original_dir = original_dir
        self.modified_dirs = modified_dirs  # lista di cartelle
        self.transform = transform
        self.pairs = []  # lista di tuple (original_path, modified_path)

        original_files = set(os.listdir(original_dir))

        # Per ogni cartella modificata e per ogni file in essa,
        # cerchiamo il corrispondente in original
        for mod_dir in self.modified_dirs:
            mod_files = os.listdir(mod_dir)
            for fname in mod_files:
                if fname in original_files:
                    orig_path = os.path.join(original_dir, fname)
                    mod_path = os.path.join(mod_dir, fname)
                    self.pairs.append((orig_path, mod_path))
                else:
                    print(f"Attenzione: {fname} in {mod_dir} non trovato in original_dir")

        print(f"Totale coppie immagini: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        orig_path, mod_path = self.pairs[idx]
        orig_img = Image.open(orig_path).convert("RGB")
        mod_img = Image.open(mod_path).convert("RGB")

        
        if self.transform:
            orig_img = self.transform(orig_img)
            mod_img = self.transform(mod_img)

        input_tensor = torch.cat([orig_img, mod_img], dim=0)

        # Ritorna anche un identificativo con nome e cartella modificata
        return input_tensor, os.path.basename(mod_path) + " - " + os.path.basename(os.path.dirname(mod_path))


# Trasformazioni immagine
transform = T.Compose([
    T.Resize((224,224)), 
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Modello CNN semplice encoder-decoder
class SimpleChangeDetector(nn.Module):
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
        x = torch.sigmoid(self.dec1(x))  # output binario 0-1

        return x

# Funzione di inferenza con visualizzazione
def infer_and_save_masks(model, dataloader, device, output_dir="output_masks"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs, filenames = batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze(1).cpu()  # shape (B, H, W)

            for i in range(outputs.size(0)):
                mask = outputs[i].numpy()
                mask_img = (mask > 0.5).astype('uint8') * 255  # binarizza

                # Converti array numpy in immagine PIL
                pil_mask = Image.fromarray(mask_img)

                # Usa filename per salvare
                save_path = os.path.join(output_dir, filenames[i].replace(" ", "_") + ".png")
                pil_mask.save(save_path)
                print(f"Salvata maschera: {save_path}")

def train_model(model, dataloader, device, epochs=5, lr=1e-3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()  # dato che l'output è binario 0-1

    for epoch in range(epochs):
        print(f"\nInizio epoch {epoch+1}/{epochs}")
        total_loss = 0
        for inputs, masks in dataloader:
            inputs = inputs.to(device)
            masks = masks.to(device)  # shape (B, 1, H, W)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

def main():
    original_dir_for_training = "data/raw/CASIA2/Authentic"
    modified_dirs_for_training = "data/raw/CASIA2/Tampered"
    mask_dir_for_training = "data/raw/CASIA2/Masks"
    original_dir_for_inference = "data/raw/val_images"
    modified_dirs_for_inference = [
        "data/processed/bbox_Kandinsky_random",
        "data/processed/bbox_Stable_Diffusion_random",
        "data/processed/bbox_Stable_Diffusion_realistic",
        "data/processed/segmentation_Kandinsky_random",
        "data/processed/segmentation_Stable_Diffusion_random",
        "data/processed/segmentation_Stable_Diffusion_realistic"
    ]


    # Dataset di training e dataloader
    triplet_file = "triplets.pkl"

    if os.path.exists(triplet_file):
        with open(triplet_file, "rb") as f:
            triplets = pickle.load(f)
        print(f"Triplette caricate da {triplet_file}")
    else:
        triplets = create_triplets_from_tampered(modified_dirs_for_training, mask_dir_for_training, original_dir_for_training)
        with open(triplet_file, "wb") as f:
            pickle.dump(triplets, f)
        print(f"Triplette generate e salvate in {triplet_file}")
    
    
    
    dataset_for_training = CASIATransformerDataset(triplets)
    train_loader = DataLoader(dataset_for_training, batch_size=2, shuffle=True)

    # Dataset di inference e dataloader
    dataset_for_inference = ImagePairDataset(original_dir_for_inference, modified_dirs_for_inference, transform=transform)
    inference_loader = DataLoader(dataset_for_inference, batch_size=2, shuffle=False)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    model = SimpleChangeDetector().to(device)

    # Addestramento
    train_model(model, train_loader, device, epochs=5, lr=1e-3)

    # Inferenza
    infer_and_save_masks(model, inference_loader, device)


    



if __name__ == "__main__":
    main()
