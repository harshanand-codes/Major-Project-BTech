import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """Combined Dice loss + Binary Cross-Entropy loss."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) raw logits
            target: (B, 1, H, W) binary mask
        """
        pred_sigmoid = torch.sigmoid(pred)

        pred_flat = pred_sigmoid.flatten(1)
        target_flat = target.flatten(1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        )
        dice_loss = 1.0 - dice.mean()

        bce_loss = F.binary_cross_entropy_with_logits(pred, target)

        return dice_loss + bce_loss


class VisionLanguageAlignmentLoss(nn.Module):
    """Cosine similarity loss between pooled visual features and text embedding."""

    def __init__(self, visual_dim, text_dim=512, proj_dim=256):
        super().__init__()
        self.visual_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(visual_dim, proj_dim),
        )
        self.text_proj = nn.Linear(text_dim, proj_dim)

    def forward(self, visual_features, text_embedding):
        """
        Args:
            visual_features: (B, C, H, W) prompt-guided features
            text_embedding: (B, text_dim) from BioMed CLIP
        Returns:
            loss: scalar
        """
        v = self.visual_proj(visual_features)
        t = self.text_proj(text_embedding)

        v = F.normalize(v, dim=-1)
        t = F.normalize(t, dim=-1)

        similarity = (v * t).sum(dim=-1)
        loss = 1.0 - similarity.mean()

        return loss


class CombinedLoss(nn.Module):
    """Overall Loss = lambda_1 * DiceBCE + lambda_4 * VL_alignment."""

    def __init__(self, visual_dim, text_dim=512, lambda_1=1.0, lambda_4=0.1):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_4 = lambda_4
        self.dice_bce = DiceBCELoss()
        self.vl_alignment = VisionLanguageAlignmentLoss(visual_dim, text_dim)

    def forward(self, pred_mask, target_mask, visual_features, text_embedding):
        loss_1 = self.dice_bce(pred_mask, target_mask)
        loss_4 = self.vl_alignment(visual_features, text_embedding)
        total = self.lambda_1 * loss_1 + self.lambda_4 * loss_4
        return total, loss_1, loss_4
