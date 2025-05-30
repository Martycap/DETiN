import torch
import torch.nn as nn



def train_model(model, dataloader, device, epochs=20, lr=1e-3, patience=5, checkpoint_path="models/CNN_masks/best_model.pth"):
    """
    Trains a binary segmentation model using the provided dataloader and saves the best model based on validation loss.
        - Uses binary cross-entropy loss for training.
        - Implements early stopping to prevent overfitting.
        - Saves the model only when the average loss improves.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        total_loss = 0
        model.train()
        
        for inputs, masks in dataloader:
            inputs, masks = inputs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint salvato: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            print(f"Nessun miglioramento per {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= patience:
            print("Early stopping attivato")
            break
