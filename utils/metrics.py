import numpy as np
from scipy.spatial.distance import directed_hausdorff


def compute_metrics(pred, target, threshold=0.5):
    """
    Compute segmentation metrics.

    Args:
        pred: (B, 1, H, W) numpy array, sigmoid probabilities
        target: (B, 1, H, W) numpy array, binary ground truth

    Returns:
        dict with dice, iou, f1, hausdorff
    """
    pred_bin = (pred > threshold).astype(np.float32)
    target = target.astype(np.float32)

    smooth = 1e-6
    batch_size = pred_bin.shape[0]

    dice_scores = []
    iou_scores = []
    hausdorff_distances = []

    for i in range(batch_size):
        p = pred_bin[i].flatten()
        t = target[i].flatten()

        intersection = (p * t).sum()
        union = p.sum() + t.sum() - intersection

        dice = (2.0 * intersection + smooth) / (p.sum() + t.sum() + smooth)
        iou = (intersection + smooth) / (union + smooth)

        dice_scores.append(dice)
        iou_scores.append(iou)

        p_2d = pred_bin[i, 0]
        t_2d = target[i, 0]

        p_coords = np.argwhere(p_2d > 0.5)
        t_coords = np.argwhere(t_2d > 0.5)

        if len(p_coords) == 0 and len(t_coords) == 0:
            hd = 0.0
        elif len(p_coords) == 0 or len(t_coords) == 0:
            hd = max(p_2d.shape)
        else:
            hd_forward = directed_hausdorff(p_coords, t_coords)[0]
            hd_backward = directed_hausdorff(t_coords, p_coords)[0]
            hd = max(hd_forward, hd_backward)

        hausdorff_distances.append(hd)

    metrics = {
        "dice": np.mean(dice_scores),
        "iou": np.mean(iou_scores),
        "f1": np.mean(dice_scores),  # F1 = Dice for binary segmentation
        "hausdorff": np.mean(hausdorff_distances),
    }
    return metrics
