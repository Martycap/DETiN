import torch, pickle, os
import torch.nn as nn
import random
from torch.utils.data import DataLoader, random_split
from features.create_tripletsCOCO import create_pairs_from_tampered
from features.visualization import set_seed, visualize_sample
from data.detin_dataset import DETINTransformerDataset
from features.create_triplets import create_pairs_from_tp_list
from models.DETIN.detin_model import prepare_model
from models.DETIN.detin_training import train
from models.DETIN.detin_inference import inference

# === Paths ===
tampered_dir_coco = "./data/processed/train/COCO"
mask_dir_coco = "./data/processed/masks/DIFF_masks"
tampered_dir_casia = "data/raw/CASIA2/Tampered"
mask_dir_casia = "data/raw/CASIA2/Masks"
tp_list_path = "data/raw/CASIA2/tp_list.txt"
casia_pairs_cache_path = "data/raw/CASIA2/pairs.pkl"
final_model_path = "models/DETIN/final_model.pth"
coco_pairs_cache_path = "data/processed/pairs_coco.pkl"


def get_dataloaders(triplets, batch_size=8, splits=(0.7, 0.15, 0.15)):
    """
    Split the dataset into training, validation, and test sets and return corresponding DataLoaders.
    """
    dataset = DETINTransformerDataset(triplets)
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
    Main pipeline for DETIN model training and inference using image/mask pairs only.
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Step 1: Load or generate COCO pairs ===
    if os.path.exists(coco_pairs_cache_path):
        with open(coco_pairs_cache_path, "rb") as f:
            coco_pairs = pickle.load(f)
        print(f"COCO pairs loaded from {coco_pairs_cache_path}")
        
        if len(coco_pairs) == 0:
            raise ValueError("COCO pairs file is empty.")
    else:
        coco_pairs = create_pairs_from_tampered(tampered_dir_coco, mask_dir_coco)
        with open(coco_pairs_cache_path, "wb") as f:
            pickle.dump(coco_pairs, f)
        print(f"COCO pairs generated and saved to {coco_pairs_cache_path}")

    # === Step 2: Load or generate CASIA pairs ===
    if os.path.exists(casia_pairs_cache_path):
        with open(casia_pairs_cache_path, "rb") as f:
            casia_pairs = pickle.load(f)
        print(f"CASIA: {len(casia_pairs)} pairs loaded from cache.")
    else:
        casia_pairs = create_pairs_from_tp_list(tp_list_path, tampered_dir_casia, mask_dir_casia)
        os.makedirs(os.path.dirname(casia_pairs_cache_path), exist_ok=True)
        with open(casia_pairs_cache_path, "wb") as f:
            pickle.dump(casia_pairs, f)
        print(f"CASIA: {len(casia_pairs)} pairs generated and saved to cache.")

    # === Step 3: Merge CASIA + COCO pairs ===
    all_pairs = casia_pairs + coco_pairs
    if not all_pairs:
        print("No image/mask pairs found. Aborting.")
        return

    print(f"Total image/mask pairs: {len(all_pairs)}")
    
    # === Step 4: Shuffle and preview a sample ===
    random.shuffle(all_pairs)
    dataset = DETINTransformerDataset(all_pairs)
    visualize_sample(dataset, index=0)

    # === Step 5: Prepare DataLoaders ===
    train_loader, val_loader, test_loader = get_dataloaders(all_pairs, batch_size=8)

    # === Step 6: Prepare the model ===
    model = prepare_model(input_channels=9, num_classes=1, pretrained=False)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # === Step 7: Training ===
    print("Starting training...")
    train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, checkpoint_dir="./models/DETIN_coco", patience=7)
    print("Training completed.")

    # === Step 8: Inference ===
    print("Starting inference on test set...")
    inference(model, test_loader, device)
    print("Inference completed.")


if __name__ == '__main__':
    main()
