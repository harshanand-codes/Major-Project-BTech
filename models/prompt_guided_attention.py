import torch
import torch.nn as nn


class PromptGuidedAttentionBlock(nn.Module):
    """Cross-attention block: text embedding guides visual feature attention."""

    def __init__(self, visual_dim, text_dim=512, num_heads=8):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, visual_dim)
        self.visual_proj = nn.Conv2d(visual_dim, visual_dim, kernel_size=1)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=visual_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(visual_dim)
        self.norm2 = nn.LayerNorm(visual_dim)

        self.ffn = nn.Sequential(
            nn.Linear(visual_dim, visual_dim * 4),
            nn.GELU(),
            nn.Linear(visual_dim * 4, visual_dim),
        )

    def forward(self, visual_feat, text_embedding):
        """
        Args:
            visual_feat: (B, C, H, W)
            text_embedding: (B, text_dim)
        Returns:
            guided_feat: (B, C, H, W)
        """
        B, C, H, W = visual_feat.shape

        v = self.visual_proj(visual_feat)
        v = v.flatten(2).permute(0, 2, 1)  # (B, H*W, C)

        t = self.text_proj(text_embedding).unsqueeze(1)  # (B, 1, C)

        # Cross-attention: visual features attend to text
        attn_out, _ = self.cross_attn(query=v, key=t, value=t)
        v = self.norm1(v + attn_out)

        v = self.norm2(v + self.ffn(v))

        guided_feat = v.permute(0, 2, 1).reshape(B, C, H, W)
        return guided_feat


class PromptGuidedAttention(nn.Module):
    """Apply prompt-guided attention to multi-scale ViT features."""

    def __init__(self, visual_dim=768, text_dim=512, num_heads=8, num_scales=4):
        super().__init__()
        self.attention_blocks = nn.ModuleList([
            PromptGuidedAttentionBlock(visual_dim, text_dim, num_heads)
            for _ in range(num_scales)
        ])

    def forward(self, features, text_embedding):
        """
        Args:
            features: dict with keys "f1".."f4", each (B, C, H, W)
            text_embedding: (B, text_dim)
        Returns:
            guided_features: dict with same keys, each (B, C, H, W)
        """
        guided = {}
        for i, block in enumerate(self.attention_blocks, start=1):
            key = f"f{i}"
            guided[key] = block(features[key], text_embedding)
        return guided
