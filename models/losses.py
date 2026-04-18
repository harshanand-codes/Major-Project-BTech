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
    """Symmetric InfoNCE between pooled visual features and text embeddings.

    Uses in-batch negatives so the loss cannot collapse to zero by mapping
    every input to the same direction (a degenerate minimum of plain cosine
    alignment). A learnable temperature (CLIP-style logit_scale) keeps the
    softmax sharp enough for useful gradients.
    """

    def __init__(self, visual_dim, text_dim=512, proj_dim=256, init_temp=0.07):
        super().__init__()
        self.visual_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(visual_dim, proj_dim),
        )
        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.logit_scale = nn.Parameter(
            torch.log(torch.tensor(1.0 / init_temp))
        )

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

        if v.size(0) < 2:
            return 1.0 - (v * t).sum(dim=-1).mean()

        scale = self.logit_scale.clamp(max=4.6052).exp()
        logits = scale * v @ t.t()
        targets = torch.arange(v.size(0), device=v.device)
        loss = 0.5 * (
            F.cross_entropy(logits, targets)
            + F.cross_entropy(logits.t(), targets)
        )
        return loss


def _info_nce(a, b, scale):
    """Symmetric InfoNCE over the batch dimension.

    Both a and b are expected to be (B, C); positives are the diagonal pairs
    (a[i], b[i]). Negatives are all off-diagonal pairs in the same batch.
    Falls back to plain cosine for B < 2 (no negatives available).
    """
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    if a.size(0) < 2:
        return 1.0 - (a * b).sum(dim=-1).mean()
    logits = scale * a @ b.t()
    targets = torch.arange(a.size(0), device=a.device)
    return 0.5 * (
        F.cross_entropy(logits, targets)
        + F.cross_entropy(logits.t(), targets)
    )


class TemporalLoss(nn.Module):
    """
    Loss 2: InfoNCE between pooled FCorrespondence and pooled FT2 features.

    Drop-in InfoNCE replacement for the original cosine alignment, preserving
    the original (f_corr, features2) pairing. Fixes the trivial-zero collapse
    of plain cosine while keeping the design intent unchanged.
    """

    def __init__(self, init_temp=0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(
            torch.log(torch.tensor(1.0 / init_temp))
        )

    def forward(self, f_corr, features2):
        """
        Args:
            f_corr: dict {"f1".."f4"} correspondence features
            features2: dict {"f1".."f4"} features from frame 2
        Returns:
            loss: scalar
        """
        scale = self.logit_scale.clamp(max=4.6052).exp()
        losses = []
        for key in f_corr:
            c = F.adaptive_avg_pool2d(f_corr[key], 1).flatten(1)
            t = F.adaptive_avg_pool2d(features2[key], 1).flatten(1)
            losses.append(_info_nce(c, t, scale))
        return torch.stack(losses).mean()


class FeatureCorrespondenceLoss(nn.Module):
    """
    Loss 3: InfoNCE between pooled FEnhanced and pooled FT1 features.

    Drop-in InfoNCE replacement for the original cosine alignment, preserving
    the original (f_enhanced, features1) pairing.
    """

    def __init__(self, init_temp=0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(
            torch.log(torch.tensor(1.0 / init_temp))
        )

    def forward(self, f_enhanced, features1):
        """
        Args:
            f_enhanced: dict {"f1".."f4"} enhanced features
            features1: dict {"f1".."f4"} original features from frame 1
        Returns:
            loss: scalar
        """
        scale = self.logit_scale.clamp(max=4.6052).exp()
        losses = []
        for key in f_enhanced:
            e = F.adaptive_avg_pool2d(f_enhanced[key], 1).flatten(1)
            t = F.adaptive_avg_pool2d(features1[key], 1).flatten(1)
            losses.append(_info_nce(e, t, scale))
        return torch.stack(losses).mean()


class CombinedLoss(nn.Module):
    """Overall Loss = l1*DiceBCE + l2*Temporal + l3*FeatureCorr + l4*VL_alignment."""

    def __init__(self, visual_dim, text_dim=512,
                 lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, lambda_4=1.0):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        self.dice_bce = DiceBCELoss()
        self.temporal = TemporalLoss()
        self.feature_corr = FeatureCorrespondenceLoss()
        self.vl_alignment = VisionLanguageAlignmentLoss(visual_dim, text_dim)

    def forward(self, pred_mask=None, target_mask=None,
                visual_features=None, text_embedding=None,
                f_corr=None, features2=None,
                f_enhanced=None, features1=None):
        """
        Computes whichever losses have the required inputs.

        Args:
            pred_mask, target_mask: for Loss 1 (Dice+BCE)
            visual_features, text_embedding: for Loss 4 (VL alignment)
            f_corr, features2: for Loss 2 (temporal InfoNCE)
            f_enhanced, features1: for Loss 3 (feature correspondence InfoNCE)

        Returns:
            total: weighted sum of all computed losses
            loss_dict: dict of individual loss values
        """
        total = torch.tensor(0.0, device=self._get_device())
        loss_dict = {}

        if pred_mask is not None and target_mask is not None:
            loss_1 = self.dice_bce(pred_mask, target_mask)
            total = total + self.lambda_1 * loss_1
            loss_dict["loss_1"] = loss_1

        if f_corr is not None and features2 is not None:
            loss_2 = self.temporal(f_corr, features2)
            total = total + self.lambda_2 * loss_2
            loss_dict["loss_2"] = loss_2

        if f_enhanced is not None and features1 is not None:
            loss_3 = self.feature_corr(f_enhanced, features1)
            total = total + self.lambda_3 * loss_3
            loss_dict["loss_3"] = loss_3

        if visual_features is not None and text_embedding is not None:
            loss_4 = self.vl_alignment(visual_features, text_embedding)
            total = total + self.lambda_4 * loss_4
            loss_dict["loss_4"] = loss_4

        loss_dict["total"] = total
        return total, loss_dict

    def _get_device(self):
        return next(self.parameters()).device
