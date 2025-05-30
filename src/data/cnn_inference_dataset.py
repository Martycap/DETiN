import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from data.make_dataset import center_crop



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
        orig_img = center_crop(orig_img)
        mod_img = Image.open(mod_path).convert("RGB")

        
        if self.transform:
            orig_img = self.transform(orig_img)
            mod_img = self.transform(mod_img)

        input_tensor = torch.cat([orig_img, mod_img], dim=0)

        # Ritorna anche un identificativo con nome e cartella modificata
        return input_tensor, os.path.basename(mod_path) + " - " + os.path.basename(os.path.dirname(mod_path))
