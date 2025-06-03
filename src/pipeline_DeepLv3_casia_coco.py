import torch, pickle, json, os, sys
from pathlib import Path
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from features.create_triplets import create_pairs_from_tp_list, create_triplets_from_tampered, extract_acronyms, show_triplet, create_pairs_COCO
from features.visualization import set_seed, visualize_sample
from data.casia_dataset import CASIATransformerDataset
from src.models.DETIN.detin_first_model import prepare_model
from models.DETIN.detin_training import train
from models.DETIN.detin_inference import inference


tampered_dir = "data/raw/CASIA2/Tampered"
mask_dir = "data/raw/CASIA2/Masks"
tp_list_path = "data/raw/CASIA2/tp_list.txt"
pairs_cache_path = "data/raw/CASIA2/pairs.pkl"
final_model_path = "models/DETIN/final_model.pth"
coco_pairs_cache_path = "data/processed/pairs_coco.pkl"



def get_dataloaders(triplets, batch_size=8, splits=(0.7, 0.15, 0.15)):
    """
    Create PyTorch DataLoaders for training, validation and testing datasets
    by splitting the CASIA dataset triplets.
    """
    dataset = CASIATransformerDataset(triplets)
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    shuffled_dataset = torch.utils.data.Subset(dataset, indices)
    
    total_len = len(shuffled_dataset)
    train_len = int(splits[0] * total_len)
    val_len = int(splits[1] * total_len)
    test_len = total_len - train_len - val_len

    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader




def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === STEP 1: Carica coppie CASIA da cache, se presenti ===
    if os.path.exists(pairs_cache_path):
        with open(pairs_cache_path, "rb") as f:
            casia_pairs = pickle.load(f)
        print(f"CASIA: {len(casia_pairs)} coppie caricate da cache.")
    else:
        # === Altrimenti, crea e salva ===
        casia_pairs = create_pairs_from_tp_list(tp_list_path, tampered_dir, mask_dir)
        os.makedirs(os.path.dirname(pairs_cache_path), exist_ok=True)
        with open(pairs_cache_path, "wb") as f:
            pickle.dump(casia_pairs, f)
        print(f"CASIA: {len(casia_pairs)} coppie create e salvate in cache.")

    # === STEP 2: Crea le coppie COCO da cache, se presenti ===
    if os.path.exists(coco_pairs_cache_path):
        with open(coco_pairs_cache_path, "rb") as f:
            coco_pairs = pickle.load(f)
        print(f"COCO: {len(casia_pairs)} coppie caricate da cache.")
    else:
        coco_pairs = create_pairs_COCO()
        os.makedirs(os.path.dirname(coco_pairs_cache_path), exist_ok=True)
        with open(coco_pairs_cache_path, "wb") as f:
            pickle.dump(coco_pairs, f)
        print(f"COCO: {len(coco_pairs)} coppie create e salvate in cache.")

    # === STEP 3: Unisci CASIA + COCO ===
    all_pairs = casia_pairs + coco_pairs
    print(f"Totale coppie unite: {len(all_pairs)}")

    if not all_pairs:
        print("Nessuna coppia valida trovata. Uscita.")
        return
    
    # Debug: mostra prima coppia
    tampered_path, mask_path = coco_pairs[0]
    print("Tampered:", tampered_path)
    print("Mask:", mask_path)

    # === STEP 4: Dataset e visualizzazione ===
    dataset = CASIATransformerDataset(all_pairs)
    visualize_sample(dataset, index=0)
    visualize_sample(dataset, index=6000)
    train_loader, val_loader, test_loader = get_dataloaders(all_pairs, batch_size=8)

    # === STEP 5: Preparazione modello ===
    model = prepare_model(input_channels=9, num_classes=1, pretrained=False)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if os.path.exists(final_model_path):
        model.load_state_dict(torch.load(final_model_path, map_location=device))
        print(f"Model loaded from {final_model_path}")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0).to(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        print("Starting training...")
        train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20)
        print("Training completed.")
        
        
    # === STEP 6: Inference ===
    print("Starting inference on test set...")
    inference(model, test_loader, device)
    print("Inference completed.")

if __name__ == '__main__':
    main()
