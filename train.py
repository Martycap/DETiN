import os
import json
import torch
import tqdm
import cv2
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.utils.create_triplets import create_triplets_from_tampered
from src.data.casia_dataset import CASIATransformerDataset, plot_image_noise_freq
import torchvision.models.segmentation as segmentation
import torchvision.transforms.functional as TF


CHECKPOINT_DIR = "models/checkpoints"
INFERENCE_DIR = "predicted_masks"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # se usi più GPU

    # Impostazioni per determinismo in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def prepare_model(input_channels=9, num_classes=1, pretrained=False):
    model = segmentation.deeplabv3_resnet50(pretrained=pretrained, num_classes=num_classes)
    old_conv = model.backbone.conv1
    model.backbone.conv1 = nn.Conv2d(
        in_channels=input_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )
    return model


def get_dataloaders(triplets, batch_size=8, splits=(0.7, 0.15, 0.15)):
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


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for x, mask in tqdm.tqdm(dataloader, desc="Training"):
        x, mask = x.to(device), mask.to(device)
        optimizer.zero_grad()
        output = model(x)['out']
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for x, mask in tqdm.tqdm(dataloader, desc="Validation"):
            x, mask = x.to(device), mask.to(device)
            output = model(x)['out']
            loss = criterion(output, mask)
            running_loss += loss.item()
    avg_loss = running_loss / len(dataloader)
    return avg_loss

def compute_iou(pred_mask, true_mask):
    pred = pred_mask.byte()
    target = true_mask.byte()
    intersection = (pred & target).float().sum((1, 2))
    union = (pred | target).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

def compute_f1(pred_mask, true_mask):
    pred = pred_mask.byte()
    target = true_mask.byte()
    tp = (pred & target).float().sum()
    fp = (pred & ~target).float().sum()
    fn = (~pred & target).float().sum()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return f1.item()

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, checkpoint_dir=CHECKPOINT_DIR, patience=3):
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved to {checkpoint_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break
    # Salvataggio finale del modello e delle metriche
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "final_model.pth"))
    with open(os.path.join(checkpoint_dir, "training_log.json"), "w") as f:
        json.dump(history, f, indent=4)



def inference(model, dataloader, device, save_dir=INFERENCE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    iou_scores = []
    f1_scores = []

    with torch.no_grad():
        for i, (x, mask) in enumerate(tqdm.tqdm(dataloader, desc="Inference")):
            x, mask = x.to(device), mask.to(device)
            outputs = model(x)['out']
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            for b in range(preds.shape[0]):
                pred_mask = preds[b, 0]
                true_mask = mask[b, 0]
                
                iou_scores.append(compute_iou(pred_mask, true_mask))
                f1_scores.append(compute_f1(pred_mask, true_mask))

                filename = f"pred_mask_{i*preds.shape[0]+b:05d}.png"
                mask_img = (pred_mask.cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(save_dir, filename), mask_img)

    # Calcolo delle metriche aggregate
    avg_iou = np.mean(iou_scores)
    avg_f1 = np.mean(f1_scores)
    print(f"Test IoU: {avg_iou:.4f}, F1-score: {avg_f1:.4f}")

    # Salvataggio dei risultati
    results = {
        "iou": avg_iou,
        "f1_score": avg_f1
    }
    with open(os.path.join(save_dir, "test_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    return results


def visualize_sample(dataset, index=0):
    x, mask = dataset[index]
    image = x[:3, :, :]
    noise = x[3:6, :, :]
    freq = x[6:9, :, :]
    plot_image_noise_freq(image, noise, freq)


def predict_single_image(model, image_tensor, device, threshold=0.5):
    """
    Esegue l'inferenza su una singola immagine pre-processata (9 canali).
    Ritorna la maschera predetta (come tensore float32 binario).
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # [1, 9, H, W]
        output = model(image_tensor)['out']
        pred = torch.sigmoid(output)
        pred = (pred > threshold).float()
    return pred.squeeze(0)  # [1, H, W] → [H, W]

def main():
    #set_seed(42)  # Fisso il seed a 42 per la riproducibilità
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    authentic_dir = 'data/raw/CASIA2/Authentic'
    tampered_dir = 'data/raw/CASIA2/Tampered'
    mask_dir = 'data/raw/CASIA2/Masks'

    
    triplets = create_triplets_from_tampered(tampered_dir, mask_dir, authentic_dir)

    if not triplets:
        print("Nessun triplet trovato.")
        return

    print(f"Triplets trovati: {len(triplets)}")  

    dataset = CASIATransformerDataset(triplets)
    visualize_sample(dataset, index=0)

    train_loader, val_loader, test_loader = get_dataloaders(triplets, batch_size=8)

    model = prepare_model(input_channels=9, num_classes=1, pretrained=False)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10)
    
    metrics = inference(model, test_loader, device, save_dir=INFERENCE_DIR)
    print(f"Risultati finali: IoU={metrics['iou']:.4f}, F1={metrics['f1_score']:.4f}")




if __name__ == "__main__":
    main()
