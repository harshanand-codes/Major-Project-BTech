import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """Single decoder stage: upsample + skip connection + conv blocks."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear",
                                     align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNetDecoder(nn.Module):
    """
    UNet-style decoder with skip connections from multi-scale encoder features.

    All ViT features are 14x14, so skip connections at each scale are projected
    to the appropriate channel width. Progressive upsampling:
      14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224
    """

    def __init__(self, encoder_dim=768, decoder_channels=(512, 256, 128, 64)):
        super().__init__()
        num_skips = len(decoder_channels) - 1
        self.skip_projs = nn.ModuleList()
        for i in range(num_skips):
            self.skip_projs.append(
                nn.Sequential(
                    nn.Conv2d(encoder_dim, decoder_channels[i], kernel_size=1),
                    nn.BatchNorm2d(decoder_channels[i]),
                    nn.ReLU(inplace=True),
                )
            )

        self.num_skips = len(decoder_channels) - 1

        self.blocks = nn.ModuleList()
        in_ch = encoder_dim
        for i, out_ch in enumerate(decoder_channels):
            skip_ch = decoder_channels[i] if i < self.num_skips else 0
            self.blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch

        self.final_conv = nn.Conv2d(decoder_channels[-1], 1, kernel_size=1)

    def forward(self, features):
        """
        Args:
            features: dict with keys "f1".."f4", each (B, encoder_dim, 14, 14)
                      f4 is deepest, f1 is shallowest
        Returns:
            mask: (B, 1, 224, 224)
        """
        # f4 (deepest) is the starting point; f3, f2, f1 are skip connections
        x = features["f4"]

        skip_keys = ["f3", "f2", "f1"]
        for i, block in enumerate(self.blocks):
            if i < len(skip_keys):
                skip = self.skip_projs[i](features[skip_keys[i]])
                # Upsample skip to match x's target size
                target_size = (x.shape[2] * 2, x.shape[3] * 2)
                skip = F.interpolate(skip, size=target_size, mode="bilinear",
                                     align_corners=False)
            else:
                skip = None
            x = block(x, skip)

        mask = self.final_conv(x)
        return mask
