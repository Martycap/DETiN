import torch 
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import numpy as np

def evaluate_model(model, dataloader, device):
    """
    Evaluates the performance of a binary segmentation model using common classification metrics.
    Predictions are binarized with a threshold of 0.5.
    All metrics are computed using flattened arrays for binary classification.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            preds = (outputs > 0.5).float()

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0).flatten().astype(int)
    all_targets = np.concatenate(all_targets, axis=0).flatten().astype(int)

    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    iou = jaccard_score(all_targets, all_preds)

    print("\nModel Evaluation:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"IoU:       {iou:.4f}")
