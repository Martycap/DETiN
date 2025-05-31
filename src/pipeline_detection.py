import torch, pickle, json, os, sys
from pathlib import Path
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from features.create_triplets import create_pairs_from_tp_list, create_triplets_from_tampered, extract_acronyms, show_triplet
from features.visualization import set_seed, visualize_sample
from data.casia_dataset import CASIATransformerDataset
from models.DETIN.detin_model import prepare_model
from models.DETIN.detin_training import train
from models.DETIN.detin_inference import inference


tampered_dir = "data/raw/CASIA2/Tampered"
mask_dir = "data/raw/CASIA2/Masks"
tp_list_path = "data/raw/CASIA2/tp_list.txt"
pairs_cache_path = "data/raw/CASIA2/pairs.pkl"



def get_dataloaders(triplets, batch_size=8, splits=(0.7, 0.15, 0.15)):
    """
    Create PyTorch DataLoaders for training, validation and testing datasets
    by splitting the CASIA dataset triplets.
    """
    dataset = CASIATransformerDataset(triplets)
    total_len = len(dataset)
    train_len = int(splits[0] * total_len)
    val_len = int(splits[1] * total_len)
    test_len = total_len - train_len - val_len

    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def main():
    """
    Main execution function:
    - Sets random seed and device.
    - Checks if acronyms JSON exists, creates it if missing.
    - Creates triplets from dataset folders.
    - Visualizes sample data.
    - Prepares model, criterion, optimizer.
    - Trains model.
    - Runs inference on test set.
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    tp_list_path = "data/raw/CASIA2/tp_list.txt"
    
    pairs = create_pairs_from_tp_list(tp_list_path,tampered_dir, mask_dir)

    print(f"Trovate {len(pairs)} coppie valide.")
    if pairs:
        tampered_path, mask_path = pairs[0]
        print("Tampered:", tampered_path)
        print("Mask:", mask_path)

        dataset = CASIATransformerDataset(pairs)
        visualize_sample(dataset, index=0)

        train_loader, val_loader, test_loader = get_dataloaders(pairs, batch_size=8)

        model = prepare_model(input_channels=9, num_classes=1, pretrained=False)
        model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        print("Starting training...")
        train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10)
        print("Training completed.")

        print("Starting inference on test set...")
        inference(model, test_loader, device)
        print("Inference completed.")



def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === STEP 1: Carica coppie da cache, se esistono ===
    if os.path.exists(pairs_cache_path):
        with open(pairs_cache_path, "rb") as f:
            pairs = pickle.load(f)
        print(f"Coppie caricate da cache ({len(pairs)} trovate)")
    else:
        # === STEP 2: Altrimenti, crea e salva ===
        pairs = create_pairs_from_tp_list(tp_list_path, tampered_dir, mask_dir)
        os.makedirs(os.path.dirname(pairs_cache_path), exist_ok=True)
        with open(pairs_cache_path, "wb") as f:
            pickle.dump(pairs, f)
        print(f"Salvate {len(pairs)} coppie valide in {pairs_cache_path}")

    if not pairs:
        print("Nessuna coppia valida trovata. Uscita.")
        return

    # Debug: mostra prima coppia
    tampered_path, mask_path = pairs[0]
    print("Tampered:", tampered_path)
    print("Mask:", mask_path)

    dataset = CASIATransformerDataset(pairs)
    visualize_sample(dataset, index=0)

    train_loader, val_loader, test_loader = get_dataloaders(pairs, batch_size=8)

    model = prepare_model(input_channels=9, num_classes=1, pretrained=False)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("Starting training...")
    train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10)
    print("Training completed.")

    print("Starting inference on test set...")
    inference(model, test_loader, device)
    print("Inference completed.")

if __name__ == '__main__':
    main()
