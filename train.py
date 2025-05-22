import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from utils.dataset_mapping import create_triplets_from_tampered, show_triplet
from src.data.casia_dataset import CASIATransformerDataset, plot_image_noise_freq
import torchvision.models.segmentation as segmentation
import tqdm
import torchvision.transforms.functional as TF
import numpy as np
import cv2


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


def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved to {checkpoint_path}")


def inference(model, dataloader, device, save_dir="predicted_masks"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i, (x, mask) in enumerate(tqdm.tqdm(dataloader, desc="Inference")):
            x = x.to(device)
            outputs = model(x)['out']
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()  # Thresholding

            for b in range(preds.shape[0]):
                pred_mask = preds[b, 0].cpu().numpy() * 255
                pred_mask = pred_mask.astype(np.uint8)
                # Salva la maschera con nome progressivo o altro sistema
                filename = f"pred_mask_{i*preds.shape[0]+b:05d}.png"
                cv2.imwrite(os.path.join(save_dir, filename), pred_mask)


def visualize_sample(dataset, index=0):
    x, mask = dataset[index]
    image = x[:3, :, :]
    noise = x[3:6, :, :]
    freq = x[6:9, :, :]
    plot_image_noise_freq(image, noise, freq)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    authentic_dir = 'data/raw/CASIA2/Authentic'
    tampered_dir = 'data/raw/CASIA2/Tampered'
    mask_dir = 'data/raw/CASIA2/Masks'

    triplets = create_triplets_from_tampered(tampered_dir, mask_dir, authentic_dir)

    if not triplets:
        print("Nessun triplet trovato.")
        return

    # Visualizza un esempio
    dataset = CASIATransformerDataset(triplets)
    visualize_sample(dataset, index=0)

    # Suddividi e ottieni dataloader
    train_loader, val_loader, test_loader = get_dataloaders(triplets)

    model = prepare_model(input_channels=9, num_classes=1, pretrained=False)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Addestramento
    train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10)

    # Inferenza su test set e salvataggio maschere
    inference(model, test_loader, device, save_dir="predicted_masks")


if __name__ == "__main__":
    main()
