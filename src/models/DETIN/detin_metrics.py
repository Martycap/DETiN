def compute_iou(pred_mask, true_mask):
    pred = pred_mask.byte()
    target = true_mask.byte()
    intersection = (pred & target).float().sum((0, 1))
    union = (pred | target).float().sum((0, 1))
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
