# Polyp Segmentation with ViT + BioMed CLIP + Cross-Frame Correspondence

A polyp segmentation pipeline for colonoscopy images that combines a Vision Transformer (ViT) encoder, prompt-guided attention via BioMed CLIP, and cross-frame temporal correspondence learning.

## Architecture

```
Input Image (224x224)
    |
ViT-Base Encoder (blocks 3, 6, 9, 12)
    |
    +--[if video frame pair]-- Cross-Frame Correspondence (CEM + CAM)
    |
Prompt-Guided Attention (BioMed CLIP text embedding)
    |
UNet-style Decoder (skip connections, progressive upsampling)
    |
Segmentation Mask (224x224)
```

**Three-step pipeline:**

1. **STEP 1** -- ViT-Base encoder extracts multi-scale features at 4 transformer block depths. A UNet-style decoder with skip connections produces the segmentation mask. Loss: Dice + BCE.

2. **STEP 2** -- Cross-frame correspondence modules (inspired by [CALICO](https://plan-lab.github.io/calico)) learn temporal consistency from colonoscopy video frame pairs. A Correspondence Extraction Module (CEM) computes cross-attention between two frames, and a Correspondence Adaptation Module (CAM) fuses the correspondence into the encoder features. Loss: Temporal + Feature Correspondence.

3. **STEP 3** -- Prompt-guided attention uses frozen BioMed CLIP text embeddings (e.g., "small round polyp") to modulate visual features via cross-attention. Loss: Vision-language cosine similarity alignment.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Datasets

**Kvasir-SEG** (segmentation): Place at `./Kvasir-SEG/` with `images/` and `masks/` subdirectories.

**LDPolypVideo** (video correspondence): Set the path in `configs/config.yaml` under `video.dataset_root`. Expected structure: `Images/<video_id>/<frame>.jpg` and `Annotations/<video_id>/<frame>.txt`.

## Prompt Generation

Before training, pre-compute text prompts from mask properties:

```bash
python -m data.prompt_generator --mask_dir ./Kvasir-SEG/masks --output ./data/prompt_cache.json
```

This generates 7 prompt categories based on polyp size and shape:
- `small/medium/large` x `round/irregular` polyp
- `normal mucosa` (for empty masks)

## Training

**Full training (Kvasir-SEG + LDPolypVideo, all 4 losses):**

```bash
python train.py --config configs/config.yaml
```

**Segmentation only (no video correspondence):**

```bash
python train.py --config configs/config.yaml --no-video
```

**Resume from checkpoint:**

```bash
python train.py --config configs/config.yaml --resume checkpoints/checkpoint_epoch_10.pth
```

Training logs all loss components per epoch (L1: Dice+BCE, L2: Temporal, L3: Feature Correspondence, L4: VL Alignment), validates on Kvasir-SEG, and evaluates on the test split after completion. Early stopping monitors validation Dice.

## Testing

**Built-in test split:**

```bash
python test.py --checkpoint checkpoints/best_model.pth
```

**External dataset:**

```bash
python test.py --checkpoint checkpoints/best_model.pth \
               --image_dir ./ETIS/images \
               --mask_dir ./ETIS/masks
```

**Save per-image results:**

```bash
python test.py --checkpoint checkpoints/best_model.pth --output results/test_results.json
```

Reports: Dice, IoU, F1, Hausdorff Distance.

### Video Testing (with cross-frame correspondence)

Evaluates on a video dataset with ground truth masks, using the full STEP 1 + STEP 2 + STEP 3 pipeline:

```bash
python test_video.py --checkpoint checkpoints/best_model.pth \
                     --image_root /path/to/video/Images \
                     --mask_root /path/to/video/Masks \
                     --frame_distance 5
```

Expected structure: `image_root/<video_id>/<frame>.jpg` and `mask_root/<video_id>/<frame>.png`.

Reports overall and per-video metrics. Save detailed results with `--output results/video_test.json`.

## Inference

### Single Image

```bash
python predict.py --checkpoint checkpoints/best_model.pth --image path/to/image.jpg
```

Saves a binary mask (`_mask.png`) and a green overlay visualization (`_overlay.png`) to `./predictions/`.

Options: `--output_dir`, `--prompt`, `--threshold`.

### Video Inference (with cross-frame correspondence)

**Frame pair** (segment frame 1 using correspondence with frame 2):

```bash
python predict_video.py --checkpoint checkpoints/best_model.pth \
                        --pair frame1.jpg frame2.jpg
```

**Full video folder** (segment every frame using a neighboring frame for correspondence):

```bash
python predict_video.py --checkpoint checkpoints/best_model.pth \
                        --video_dir /path/to/video/frames/ \
                        --frame_distance 5
```

Saves `_mask.png` and `_overlay.png` for each frame to `./predictions_video/`.

Options: `--output_dir`, `--prompt`, `--threshold`, `--frame_distance`.

## Configuration

All hyperparameters are in `configs/config.yaml`:

| Section | Key Parameters |
|---|---|
| `model` | `vit_model`, `feature_blocks`, `encoder_dim`, `decoder_channels` |
| `correspondence` | `fusion_mode` (add/multiply), `num_heads` |
| `loss` | `lambda_1` (Dice+BCE), `lambda_2` (Temporal), `lambda_3` (Feature Corr.), `lambda_4` (VL) |
| `data` | `dataset_root`, `train_ratio`, `val_ratio`, `test_ratio`, `seed` |
| `video` | `dataset_root`, `frame_distance_min/max`, `samples_per_epoch` |
| `training` | `batch_size`, `lr`, `epochs`, `early_stopping_patience` |

## Project Structure

```
project/
├── configs/config.yaml
├── models/
│   ├── vit_encoder.py                  # ViT-Base with multi-scale feature taps
│   ├── cross_frame_correspondence.py   # CEM + CAM modules
│   ├── prompt_guided_attention.py      # BioMed CLIP text-guided cross-attention
│   ├── decoder.py                      # UNet-style decoder
│   ├── losses.py                       # Dice+BCE, Temporal, Feature Corr., VL alignment
│   └── segmentation_model.py           # Full pipeline
├── data/
│   ├── dataset.py                      # Kvasir-SEG loader + mixed dataloaders
│   ├── video_dataset.py                # LDPolypVideo frame-pair loader
│   ├── transforms.py                   # Image/mask augmentations
│   ├── prompt_generator.py             # Pre-compute prompts from masks
│   └── prompt_cache.json               # Cached prompts
├── utils/
│   └── metrics.py                      # Dice, IoU, Hausdorff, F1
├── train.py                            # Training with mixed batches
├── test.py                             # Evaluation on image test datasets
├── test_video.py                       # Evaluation on video test datasets
├── predict.py                          # Single-image inference
├── predict_video.py                    # Video inference (frame pairs / full video)
└── requirements.txt
```

## Overall Loss

```
Loss = lambda_1 * (Dice + BCE)
     + lambda_2 * Temporal Loss
     + lambda_3 * Feature Correspondence Loss
     + lambda_4 * VL Alignment Loss
```

When training without video (`--no-video`), only Loss 1 and Loss 4 are active.
