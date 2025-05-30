import os
import torch
import pickle
import torchvision.transforms as T
from torch.utils.data import DataLoader
from data.cnn_inference_dataset import ImagePairDataset
from data.cnn_training_dataset import CASIADatasetCNN
from models.cnn_mask import ChangeDetector
from models.cnn_eval import evaluate_model
from models.cnn_iner import infer_and_save_masks
from models.cnn_training import train_model
from features.create_triplets import create_triplets_from_tampered

# Paths
original_dir_for_training = "data/raw/CASIA2/Authentic"
modified_dirs_for_training = "data/raw/CASIA2/Tampered"
mask_dir_for_training = "data/raw/CASIA2/Masks"
original_dir_for_inference = "data/raw/COCO"
modified_dirs_for_inference = [
    "data/processed/train/bbox_Kandinsky_random",
    "data/processed/train/bbox_Stable_Diffusion_random",
    "data/processed/train/bbox_Stable_Diffusion_realistic",
    "data/processed/train/random_box_Kandinsky_random",
    "data/processed/train/random_box_Stable_Diffusion_random",
    "data/processed/train/random_box_Stable_Diffusion_realistic",
    "data/processed/train/segmentation_Kandinsky_random",
    "data/processed/train/segmentation_Stable_Diffusion_random",
    "data/processed/train/segmentation_Stable_Diffusion_realistic"
]
triplet_file = "data/raw/CASIA2/triplets.pkl"
best_model_path = "models/CNN_masks/best_model.pth"
infer_masks = "./data/processed/masks/CNN_masks"

def main():
    # Select device
    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.backends.mps.is_available():
        device_name = "mps"
    device = torch.device(device=device_name)

    # Load or generate triplets
    if os.path.exists(triplet_file):
        with open(triplet_file, "rb") as f:
            triplets = pickle.load(f)
        print(f"Triplets loaded from {triplet_file}")
        
        if len(triplets) == 0:
            raise ValueError("The triplets.pkl file is empty. Check triplet generation.")
    else:
        triplets = create_triplets_from_tampered(modified_dirs_for_training, mask_dir_for_training, original_dir_for_training)
        with open(triplet_file, "wb") as f:
            pickle.dump(triplets, f)
        print(f"Triplets generated and saved to {triplet_file}")
        
    # Prepare training dataset and DataLoader
    dataset_for_training = CASIADatasetCNN(triplets)
    train_loader = DataLoader(dataset_for_training, batch_size=2, shuffle=True)

    # Define image transforms
    transform = T.Compose([
        T.Resize((256, 256)), 
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    # Prepare inference dataset and DataLoader
    dataset_for_inference = ImagePairDataset(original_dir_for_inference, modified_dirs_for_inference, transform=transform)
    inference_loader = DataLoader(dataset_for_inference, batch_size=2, shuffle=False)
    
    # Initialize model
    CNN_masks = ChangeDetector().to(device)

    # Load best model if it exists, otherwise train a new one
    if os.path.exists(best_model_path):
        CNN_masks.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Best model loaded from {best_model_path}")
    else:
        print("Model not found. Starting training...")
        train_model(CNN_masks, train_loader, device, epochs=20, lr=1e-3, patience=5)
        if os.path.exists(best_model_path):
            CNN_masks.load_state_dict(torch.load(best_model_path, map_location=device))
            print("Best model loaded after training.")
        else:
            print("Warning: model was not saved correctly.")
            return

    # Evaluate model on training data
    evaluate_model(CNN_masks, train_loader, device)


    if os.path.isdir(infer_masks) and not os.listdir(infer_masks):
        print("Run inference and save the predicted masks")
        infer_and_save_masks(CNN_masks, inference_loader, device, infer_masks)
    else:
        print("The masks have been previously created.")


if __name__ == "__main__":
    main()
