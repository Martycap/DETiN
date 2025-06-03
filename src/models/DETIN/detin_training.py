import torch, tqdm, os, json


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for x, mask, f in tqdm.tqdm(dataloader, desc="Training"):
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
        for x, mask, f in tqdm.tqdm(dataloader, desc="Validation"):
            x, mask = x.to(device), mask.to(device)
            output = model(x)['out']
            loss = criterion(output, mask)
            running_loss += loss.item()
    avg_loss = running_loss / len(dataloader)
    return avg_loss

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100, checkpoint_dir="models/DETIN", patience=5):
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

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "final_model.pth"))
    with open(os.path.join(checkpoint_dir, "training_log.json"), "w") as f:
        json.dump(history, f, indent=4)
