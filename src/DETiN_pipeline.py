import torch, pickle, os, random
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from features.create_pairs import create_coco_pairs
from features.create_pairs import create_casia_pairs
from features.visualization import set_seed, visualize_sample
from data.DETiN_dataset import DETiNTransformerDataset
from models.DETiN.detin_model import DETiN
from models.DETiN.detin_training import train
from models.DETiN.detin_inference import inference

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# === Paths ===
tampered_dir_coco = "./data/processed/inpainted"
mask_dir_coco = "./data/processed/masks/CNN_masks"
tampered_dir_casia = "data/raw/CASIA2/Tampered"
mask_dir_casia = "data/raw/CASIA2/Masks"
tp_list_path = "data/raw/CASIA2/tp_list.txt"
casia_pairs_cache_path = "data/util/pairs_casia.pkl"
coco_pairs_cache_path = "data/util/pairs_coco.pkl"
final_model_path = "models/DETIN/final_model.pth"

def load_coco_pairs():
    """
    Loads COCO pairs.
    """
    if os.path.exists(coco_pairs_cache_path):
        with open(coco_pairs_cache_path, "rb") as f:
            coco_pairs = pickle.load(f)
        print(f"COCO pairs loaded from {coco_pairs_cache_path}")
        if len(coco_pairs) == 0:
            raise ValueError("COCO pairs file is empty.")
    else:
        coco_pairs = create_coco_pairs(tampered_dir_coco, mask_dir_coco)
        with open(coco_pairs_cache_path, "wb") as f:
            pickle.dump(coco_pairs, f)
        print(f"COCO pairs generated and saved to {coco_pairs_cache_path}")
    return coco_pairs

def load_casia_pairs():
    """
    Loads CASIA pairs.
    """
    if os.path.exists(casia_pairs_cache_path):
        with open(casia_pairs_cache_path, "rb") as f:
            casia_pairs = pickle.load(f)
        print(f"CASIA: {len(casia_pairs)} pairs loaded from cache.")
    else:
        casia_pairs = create_casia_pairs(tp_list_path, tampered_dir_casia, mask_dir_casia)
        os.makedirs(os.path.dirname(casia_pairs_cache_path), exist_ok=True)
        with open(casia_pairs_cache_path, "wb") as f:
            pickle.dump(casia_pairs, f)
        print(f"CASIA: {len(casia_pairs)} pairs generated and saved to cache.")
    return casia_pairs


def get_dataloaders(pairs, batch_size=8, splits=(0.7, 0.15, 0.15)):
    """
    Split the dataset into training, validation, and test sets and 
    return corresponding DataLoaders.
    """

    dataset = DETiNTransformerDataset(pairs)
    total_len = len(dataset)
    train_len = int(splits[0] * total_len)
    val_len = int(splits[1] * total_len)
    test_len = total_len - train_len - val_len
    train_set, val_set, test_set = random_split(
        dataset, [train_len, val_len, test_len]
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def detection_pipeline(use_coco = True, use_casia = True):
    """
    Main pipeline for DETIN model.
    """

    set_seed(42)
    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.backends.mps.is_available():
        device_name = "mps"
    device = torch.device(device=device_name)

    # === Step 1: Pairs load ===
    all_pairs = []
    if use_coco:
        all_pairs += load_coco_pairs()
    if use_casia:
        all_pairs += load_casia_pairs()

    if not all_pairs:
        print("No image/mask pairs found. Aborting.")
        return

    print(f"Total image/mask pairs: {len(all_pairs)}")

    # === Step 4: Shuffle and preview a sample ===
    random.shuffle(all_pairs)
    dataset = DETiNTransformerDataset(all_pairs)
    visualize_sample(dataset, index=0)

    # === Step 5: Prepare DataLoaders ===
    train_loader, val_loader, test_loader = get_dataloaders(
        all_pairs, batch_size=8
    )

    # === Step 6: Prepare DETiN ===
    model = DETiN(num_classes=1, pretrained=False).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=1e-5
    )

    # === Step 7: Training ===
    print("Starting training...")
    train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=50,
        checkpoint_dir="./models/DETIN",
        patience=7,
    )
    print("Training completed.")

    # === Step 8: Inference ===
    print("Starting inference on test set...")
    inference(model, test_loader, device)
    print("Inference completed.")

if __name__ == "__main__":
    detection_pipeline()