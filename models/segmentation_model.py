import torch
import torch.nn as nn
import open_clip

from .vit_encoder import ViTEncoder
from .prompt_guided_attention import PromptGuidedAttention
from .decoder import UNetDecoder
from .cross_frame_correspondence import CrossFrameCorrespondence


class BioMedCLIPTextEncoder(nn.Module):
    """Frozen BioMed CLIP text encoder for prompt embeddings."""

    def __init__(self):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        self.tokenizer = open_clip.get_tokenizer(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        self.clip_model = model

        for param in self.clip_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, text_prompts):
        """
        Args:
            text_prompts: list of strings, length B
        Returns:
            text_features: (B, 512) normalized text embeddings
        """
        tokens = self.tokenizer(text_prompts)
        tokens = tokens.to(next(self.clip_model.parameters()).device)
        text_features = self.clip_model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.float()


class PolypSegmentationModel(nn.Module):
    """
    Full pipeline:
      - Single image: ViT Encoder -> Prompt-Guided Attention -> Decoder
      - Frame pair:   ViT Encoder(x2) -> CrossFrameCorrespondence -> Prompt-Guided Attention -> Decoder
    """

    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg["model"]

        self.encoder = ViTEncoder(
            model_name=model_cfg["vit_model"],
            pretrained=model_cfg["vit_pretrained"],
            feature_blocks=tuple(model_cfg["feature_blocks"]),
        )

        self.text_encoder = BioMedCLIPTextEncoder()

        self.prompt_attention = PromptGuidedAttention(
            visual_dim=model_cfg["encoder_dim"],
            text_dim=model_cfg["text_dim"],
            num_heads=model_cfg["num_heads"],
            num_scales=len(model_cfg["feature_blocks"]),
        )

        self.decoder = UNetDecoder(
            encoder_dim=model_cfg["encoder_dim"],
            decoder_channels=tuple(model_cfg["decoder_channels"]),
        )

        corr_cfg = cfg.get("correspondence", {})
        self.correspondence = CrossFrameCorrespondence(
            dim=model_cfg["encoder_dim"],
            num_heads=corr_cfg.get("num_heads", model_cfg["num_heads"]),
            num_scales=len(model_cfg["feature_blocks"]),
            fusion_mode=corr_cfg.get("fusion_mode", "add"),
        )

    def forward(self, images, text_prompts, images2=None):
        """
        Args:
            images: (B, 3, 224, 224) primary images
            text_prompts: list of strings, length B
            images2: (B, 3, 224, 224) second frame (optional, for correspondence)
        Returns:
            pred_mask: (B, 1, 224, 224) raw logits
            guided_features: dict of prompt-guided features (for VL loss)
            text_embedding: (B, 512) text features (for VL loss)
            corr_outputs: dict with f_corr, f_enhanced, features1, features2
                          (None if no frame pair)
        """
        features1 = self.encoder(images)
        text_embedding = self.text_encoder(text_prompts)

        corr_outputs = None
        if images2 is not None:
            features2 = self.encoder(images2)
            f_corr, f_enhanced = self.correspondence(features1, features2)
            corr_outputs = {
                "f_corr": f_corr,
                "f_enhanced": f_enhanced,
                "features1": features1,
                "features2": features2,
            }
            features_for_attention = f_enhanced
        else:
            features_for_attention = features1

        guided_features = self.prompt_attention(features_for_attention, text_embedding)
        pred_mask = self.decoder(guided_features)

        return pred_mask, guided_features, text_embedding, corr_outputs
