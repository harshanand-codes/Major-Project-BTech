# Polyp Segmentation with Encoder + BioMed CLIP + Cross-Frame Correspondence

A polyp segmentation pipeline for colonoscopy images that combines a encoder like ViT / Mamba / Dinov2 / CNNs like Resnet, VGG, etc., prompt-guided attention via BioMed CLIP, and cross-frame temporal correspondence learning.

## Results and Training logs

See the directory [report/](./report) for detailed results, including test metrics and training logs for various encoders with and without proposed modules (CEM + CAM + Prompt Guided Attention).

## Architecture

```
Input Image (224x224)
    |
Encoder (blocks 3, 6, 9, 12) (ViT / Mamba / Dinov2 / CNNs like Resnet, VGG, etc.)
    |
Cross-Frame Correspondence (CEM + CAM)
    |
Prompt-Guided Attention (BioMed CLIP text embedding)
    |
UNet-style Decoder (skip connections, progressive upsampling)
    |
Segmentation Mask (224x224)
```

**Three-step pipeline:**

1. **STEP 1** -- Encoders like ViT / Mamba / Dinov2 / CNNs like Resnet, VGG, etc. extracts multi-scale features at 4 transformer block depths. A UNet-style decoder with skip connections produces the segmentation mask. Loss: Dice + BCE.

2. **STEP 2** -- Cross-frame correspondence modules (inspired by [CALICO](https://plan-lab.github.io/calico)) learn temporal consistency from colonoscopy video frame pairs. A Correspondence Extraction Module (CEM) computes cross-attention between two frames, and a Correspondence Adaptation Module (CAM) fuses the correspondence into the encoder features. Loss: Temporal + Feature Correspondence.

3. **STEP 3** -- Prompt-guided attention uses frozen BioMed CLIP text embeddings (e.g., "small round polyp") to modulate visual features via cross-attention. Loss: Vision-language alignment loss (Symmetric Cross Entropy Contrastive Loss).

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Datasets

**Image segmentation dataset**: Place at `./Kvasir-SEG/` (or any path set via `data.dataset_root`) with `images/` and `masks/` subdirectories.

**Video segmentation dataset** (video correspondence + segmentation): Set the path in `configs/config.yaml` under `video.dataset_root`. Expected structure (one folder per sequence, with per-frame images and binary masks):

```
<video.dataset_root>/
└── <sequence_id>/
    ├── images/<frame>.jpg
    └── masks/<frame>.jpg   # binary mask, same filename as the image
```

Sequences are split into train/val/test according to `video.train_ratio` / `val_ratio` / `test_ratio` (seeded by `video.seed`); all valid frame pairs in the chosen sequences are then enumerated.

## Prompt Generation

Before training, pre-compute text prompts from mask properties for both datasets in one shot. The script reads paths and thresholds directly from the config:

```bash
python -m data.prompt_generator --config configs/config.yaml
```

This writes:
- `data.prompt_cache` (default `./data/prompt_cache.json`) for the image dataset
- `video.prompt_cache` (default `./data/video_prompt_cache.json`) for the video dataset

It generates 7 prompt categories based on polyp size and shape:
- `small/medium/large` x `round/irregular` polyp
- `normal mucosa` (for empty masks)

## Training

**Full training (image segmentation + video segmentation, all 4 losses):**

```bash
python train.py --config configs/config.yaml
```

**Resume from checkpoint:**

```bash
python train.py --config configs/config.yaml --resume checkpoints/checkpoint_epoch_10.pth
```

Training logs all loss components per epoch (L1: Dice+BCE, L2: Temporal, L3: Feature Correspondence, L4: VL Alignment) and validates on val split each epoch. After training, the best checkpoint is evaluated on the image and video test splits. Per-epoch metrics are appended to `<save_dir>/metrics.jsonl` for later analysis. Early stopping monitors validation Dice.

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

Evaluates on a video dataset with ground-truth masks, using the full STEP 1 + STEP 2 + STEP 3 pipeline. Two modes are supported:

**Config-based test split** (default -- uses the held-out sequences from `configs/config.yaml`):

```bash
python test_video.py --checkpoint checkpoints/best_model.pth \
                     --config configs/config.yaml
```

**Custom single sequence** (a single `images/` + `masks/` directory):

```bash
python test_video.py --checkpoint checkpoints/best_model.pth \
                     --image_dir /path/to/sequence/images \
                     --mask_dir  /path/to/sequence/masks \
                     --frame_distance 5
```

Reports Dice, IoU, F1, Hausdorff, and the full L1+L2+L3+L4 loss breakdown. Save full results with `--output results/video_test.json`.

## Inference

### Single Image

```bash
python predict.py --checkpoint checkpoints/best_model.pth --image path/to/image.jpg
```

Saves a binary mask (`_mask.png`) and a green overlay visualization (`_overlay.png`) to `./predictions/`.

Options: `--output_dir`, `--prompt`, `--threshold`.

### Video Inference (with cross-frame correspondence)

**Frame pair** (segment frame 1 using correspondence with previous neighbouring frames):

```bash
python predict_video.py --checkpoint checkpoints/best_model.pth \
                        --pair frame1.jpg frame2.jpg
```

**Full video folder** (segment every frame using a previous neighboring frames for correspondence):

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
| `data` | `dataset_root`, `prompt_cache`, `train_ratio`, `val_ratio`, `test_ratio`, `seed` |
| `video` | `dataset_root`, `prompt_cache`, `frame_distance_min/max`, `batch_size`, `train_ratio`, `val_ratio`, `test_ratio`, `seed` |
| `training` | `batch_size`, `lr`, `epochs`, `early_stopping_patience`, `save_dir` |
| `prompt` | `size_thresholds.small/large`, `circularity_threshold` |

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
│   ├── dataset.py                      # ImageSegDataset loader + mixed dataloaders
│   ├── video_dataset.py                # VideoSegDataset frame-pair loader (img + mask)
│   ├── transforms.py                   # Image/mask augmentations
│   ├── prompt_generator.py             # Pre-compute prompts from masks (image + video)
│   ├── prompt_cache.json               # Cached image-dataset prompts
│   └── video_prompt_cache.json         # Cached video-dataset prompts
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


