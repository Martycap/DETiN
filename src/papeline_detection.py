import torch, pickle, json, os, sys
from pathlib import Path
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from features.create_triplets import create_triplets_from_tampered, extract_acronyms, show_triplet
from features.visualization import set_seed, visualize_sample
from data.casia_dataset import CASIATransformerDataset
from models.DETIN.detin_model import prepare_model
from models.DETIN.detin_training import train
from models.DETIN.detin_inference import inference

acronyms_path = Path("data/raw/CASIA2/list_acronyms.json")
authentic_dir = "data/raw/CASIA2/Authentic"
tampered_dir = "data/raw/CASIA2/Tampered"
mask_dir = "data/raw/CASIA2/Masks"
triplet_file = "data/raw/CASIA2/triplets.pkl"


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
    
    if acronyms_path.exists():
        print("Acronyms have already been generated.")
    else:
        input_filename = "data/raw/CASIA2/au_list.txt"
        output_filename = "data/raw/CASIA2/list_acronyms.json"
        acronyms_list = extract_acronyms(input_filename)
        with open(output_filename, 'w') as f:
            json.dump(acronyms_list, f)
        print("Acronyms extracted and saved.")

    # Load or generate triplets
    if os.path.exists(triplet_file):
        with open(triplet_file, "rb") as f:
            triplets = pickle.load(f)
        print(f"Triplets loaded from {triplet_file}")
        
        if len(triplets) == 0:
            raise ValueError("The triplets.pkl file is empty. Check triplet generation.")
    else:
        triplets = create_triplets_from_tampered(tampered_dir, mask_dir, authentic_dir)
        with open(triplet_file, "wb") as f:
            pickle.dump(triplets, f)
        print(f"Triplets generated and saved to {triplet_file}")

    print(f"Total triplets found: {len(triplets)}")

    if not triplets:
        print("No triplets found.")
        return
    else:
        print(f"Total triplets found: {len(triplets)}")
        show_triplet(*triplets[0])

        dataset = CASIATransformerDataset(triplets)
        visualize_sample(dataset, index=0)

        train_loader, val_loader, test_loader = get_dataloaders(triplets, batch_size=8)

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


# if __name__ == '__main__':
#     main()
