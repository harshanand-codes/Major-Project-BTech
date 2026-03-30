import torch
import torch.nn as nn
import timm


class ViTEncoder(nn.Module):
    """ViT-Base encoder with intermediate feature extraction at specified blocks."""

    def __init__(self, model_name="vit_base_patch16_224", pretrained=True,
                 feature_blocks=(3, 6, 9, 12)):
        super().__init__()
        self.feature_blocks = feature_blocks
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        self.embed_dim = self.vit.embed_dim
        self.patch_size = self.vit.patch_embed.patch_size[0]

        self._features = {}
        self._register_hooks()

    def _register_hooks(self):
        for idx in self.feature_blocks:
            block = self.vit.blocks[idx - 1]
            block.register_forward_hook(self._make_hook(idx))

    def _make_hook(self, block_idx):
        def hook(module, input, output):
            self._features[block_idx] = output
        return hook

    def forward(self, x):
        B = x.shape[0]
        self._features = {}

        _ = self.vit.forward_features(x)

        num_patches_side = x.shape[-1] // self.patch_size

        features = {}
        for i, idx in enumerate(self.feature_blocks, start=1):
            feat = self._features[idx]
            # Remove CLS token: (B, 1+N, D) -> (B, N, D)
            feat = feat[:, 1:, :]
            feat = feat.permute(0, 2, 1).reshape(
                B, self.embed_dim, num_patches_side, num_patches_side
            )
            features[f"f{i}"] = feat

        return features
