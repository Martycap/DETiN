import torch
import tqdm
import os
import json
import cv2
import numpy as np

from models.detin_metrics import compute_f1, compute_iou

def inference(model, dataloader, device, save_dir="eval/DETIN_inference"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    iou_scores = []
    f1_scores = []

    with torch.no_grad():
        for i, (x, mask, filename) in enumerate(tqdm.tqdm(dataloader, desc="Inference")):
            x, mask, filenames = x.to(device), mask.to(device), filename
            outputs = model(x)['out']
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            for b in range(preds.shape[0]):
                pred_mask = preds[b, 0]
                true_mask = mask[b, 0]
                
                iou_scores.append(compute_iou(pred_mask, true_mask))
                f1_scores.append(compute_f1(pred_mask, true_mask))
                filename_no_ext = os.path.splitext(filenames[b])[0]
                filename = f"pred_mask_{filename_no_ext}.png"
                mask_img = (pred_mask.cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(save_dir, filename), mask_img)

    avg_iou = np.mean(iou_scores)
    avg_f1 = np.mean(f1_scores)
    print(f"Test IoU: {avg_iou:.4f}, F1-score: {avg_f1:.4f}")

    results = {
        "iou": avg_iou,
        "f1_score": avg_f1
    }
    with open(os.path.join(save_dir, "test_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    return results

def predict_single_image(model, image_tensor, device, threshold=0.5):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        output = model(image_tensor)['out']
        pred = torch.sigmoid(output)
        pred = (pred > threshold).float()
    return pred.squeeze(0)
