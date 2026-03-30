import torch
import torch.nn as nn


class CorrespondenceExtractionModule(nn.Module):
    """
    CEM: Cross-attention between features from two frames.
    FCorrespondence = CrossAttention(Q=FT1, K=FT2, V=FT2)
    """

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, ft1, ft2):
        """
        Args:
            ft1: (B, C, H, W) features from frame 1
            ft2: (B, C, H, W) features from frame 2
        Returns:
            f_corr: (B, C, H, W) correspondence features
        """
        B, C, H, W = ft1.shape

        q = self.proj(ft1).flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        k = ft2.flatten(2).permute(0, 2, 1)
        v = ft2.flatten(2).permute(0, 2, 1)

        attn_out, _ = self.cross_attn(query=q, key=k, value=v)
        attn_out = self.norm(attn_out + q)

        f_corr = attn_out.permute(0, 2, 1).reshape(B, C, H, W)
        return f_corr


class CorrespondenceAdaptationModule(nn.Module):
    """
    CAM: Fuses FT1 with FCorrespondence to produce FEnhanced.
    Supports addition or element-wise multiplication fusion.
    """

    def __init__(self, dim, num_heads=8, fusion_mode="add"):
        super().__init__()
        self.fusion_mode = fusion_mode

        if fusion_mode == "multiply":
            self.gate = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.Sigmoid(),
            )

        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, ft1, f_corr):
        """
        Args:
            ft1: (B, C, H, W) original features from frame 1
            f_corr: (B, C, H, W) correspondence features from CEM
        Returns:
            f_enhanced: (B, C, H, W) enhanced features
        """
        B, C, H, W = ft1.shape

        if self.fusion_mode == "multiply":
            fused = ft1 * self.gate(f_corr)
        else:
            fused = ft1 + f_corr

        x = fused.flatten(2).permute(0, 2, 1)  # (B, H*W, C)

        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))

        f_enhanced = x.permute(0, 2, 1).reshape(B, C, H, W)
        return f_enhanced


class CrossFrameCorrespondence(nn.Module):
    """
    Chains CEM and CAM for each feature scale (f1-f4).
    """

    def __init__(self, dim=768, num_heads=8, num_scales=4, fusion_mode="add"):
        super().__init__()
        self.cem_blocks = nn.ModuleList([
            CorrespondenceExtractionModule(dim, num_heads)
            for _ in range(num_scales)
        ])
        self.cam_blocks = nn.ModuleList([
            CorrespondenceAdaptationModule(dim, num_heads, fusion_mode)
            for _ in range(num_scales)
        ])

    def forward(self, features1, features2):
        """
        Args:
            features1: dict {"f1".."f4"}, each (B, C, H, W) from frame 1
            features2: dict {"f1".."f4"}, each (B, C, H, W) from frame 2
        Returns:
            f_corr: dict {"f1".."f4"} correspondence features
            f_enhanced: dict {"f1".."f4"} enhanced features
        """
        f_corr = {}
        f_enhanced = {}

        for i, (cem, cam) in enumerate(zip(self.cem_blocks, self.cam_blocks), start=1):
            key = f"f{i}"
            corr = cem(features1[key], features2[key])
            enhanced = cam(features1[key], corr)
            f_corr[key] = corr
            f_enhanced[key] = enhanced

        return f_corr, f_enhanced
