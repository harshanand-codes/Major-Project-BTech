"""
Video test: evaluates segmentation on video frame pairs.

Two modes:
  1. Config-based test split (default):
       python test_video.py --checkpoint best_model.pth

  2. Custom single-sequence evaluation:
       python test_video.py --checkpoint best_model.pth \
           --image_dir /path/to/images --mask_dir /path/to/masks
"""

import argparse
import json
import os

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from tqdm import tqdm

from data.video_dataset import VideoSegDataset, video_collate_fn, _numeric_sort_key
from models.segmentation_model import PolypSegmentationModel
from models.losses import CombinedLoss
from utils.metrics import compute_metrics


class SingleSequenceDataset(Dataset):
    """Evaluate all consecutive frame pairs from a single image/mask directory."""

    def __init__(self, image_dir, mask_dir, image_size=224, frame_distance=5):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        all_frames = sorted(
            [f for f in os.listdir(image_dir)
             if f.lower().endswith((".jpg", ".png", ".jpeg"))],
            key=_numeric_sort_key,
        )

        mask_files = set(os.listdir(mask_dir))
        self.frames = []
        for f in all_frames:
            stem = os.path.splitext(f)[0]
            if (f"{stem}.jpg" in mask_files or f"{stem}.png" in mask_files
                    or f"{stem}.jpeg" in mask_files):
                self.frames.append(f)

        self.pairs = []
        for i in range(len(self.frames)):
            j = min(i + frame_distance, len(self.frames) - 1)
            if j != i:
                self.pairs.append((i, j))

        self.image_normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def _find_mask(self, frame_name):
        stem = os.path.splitext(frame_name)[0]
        for ext in (".jpg", ".png", ".jpeg"):
            path = os.path.join(self.mask_dir, stem + ext)
            if os.path.exists(path):
                return path
        return None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx1, idx2 = self.pairs[idx]

        img1 = Image.open(os.path.join(self.image_dir, self.frames[idx1])).convert("RGB")
        img2 = Image.open(os.path.join(self.image_dir, self.frames[idx2])).convert("RGB")
        mask1 = Image.open(self._find_mask(self.frames[idx1])).convert("L")
        mask2 = Image.open(self._find_mask(self.frames[idx2])).convert("L")

        img1 = img1.resize((self.image_size, self.image_size), Image.BILINEAR)
        img2 = img2.resize((self.image_size, self.image_size), Image.BILINEAR)

        img1 = self.image_normalize(TF.to_tensor(img1))
        img2 = self.image_normalize(TF.to_tensor(img2))

        mask1 = mask1.resize((self.image_size, self.image_size), Image.NEAREST)
        mask1_np = (np.array(mask1.convert("L")) > 128).astype(np.float32)
        mask1 = torch.from_numpy(mask1_np).unsqueeze(0)

        mask2 = mask2.resize((self.image_size, self.image_size), Image.NEAREST)
        mask2_np = (np.array(mask2.convert("L")) > 128).astype(np.float32)
        mask2 = torch.from_numpy(mask2_np).unsqueeze(0)

        return img1, img2, mask1, mask2, "polyp"


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]

    model = PolypSegmentationModel(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint (epoch {checkpoint.get('epoch', '?')}, "
          f"Dice {checkpoint.get('best_dice', 'N/A')})")

    return model, cfg


@torch.no_grad()
def evaluate_video_test(model, loader, criterion, device):
    model.eval()
    totals = {"loss": 0, "loss_1": 0, "loss_2": 0, "loss_3": 0, "loss_4": 0}
    num_batches = 0

    all_preds = []
    all_targets = []

    for vid_imgs1, vid_imgs2, vid_masks1, vid_masks2, vid_prompts in tqdm(
        loader, desc="VideoTest", leave=False
    ):
        vid_imgs1 = vid_imgs1.to(device)
        vid_imgs2 = vid_imgs2.to(device)
        vid_masks1 = vid_masks1.to(device)

        pred_mask, guided_features, text_embedding, corr_outputs = model(
            vid_imgs1, vid_prompts, images2=vid_imgs2
        )

        loss, loss_dict = criterion(
            pred_mask=pred_mask,
            target_mask=vid_masks1,
            visual_features=guided_features["f4"],
            text_embedding=text_embedding,
            f_corr=corr_outputs["f_corr"],
            features2=corr_outputs["features2"],
            f_enhanced=corr_outputs["f_enhanced"],
            features1=corr_outputs["features1"],
        )

        totals["loss"] += loss.item()
        for k in ["loss_1", "loss_2", "loss_3", "loss_4"]:
            if k in loss_dict:
                totals[k] += loss_dict[k].item()
        num_batches += 1

        pred_sigmoid = torch.sigmoid(pred_mask).cpu().numpy()
        all_preds.append(pred_sigmoid)
        all_targets.append(vid_masks1.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metrics = compute_metrics(all_preds, all_targets)

    for k, v in totals.items():
        metrics[k] = v / max(num_batches, 1)

    return metrics


def _print_results(metrics, label, count):
    print("\n" + "=" * 50)
    print(f"VIDEO TEST RESULTS - {label} ({count} pairs)")
    print("=" * 50)
    print(f"  Dice:              {metrics['dice']:.4f}")
    print(f"  IoU:               {metrics['iou']:.4f}")
    print(f"  F1:                {metrics['f1']:.4f}")
    print(f"  Hausdorff Dist:    {metrics['hausdorff']:.2f}")
    print(f"  Loss:              {metrics['loss']:.4f}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Test segmentation on video data"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Config file (for config-based test split mode)")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Image directory for custom single-sequence evaluation")
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="Mask directory for custom single-sequence evaluation")
    parser.add_argument("--frame_distance", type=int, default=5,
                        help="Frame distance for neighbor pairing (custom mode)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON")
    args = parser.parse_args()

    custom_mode = args.image_dir is not None and args.mask_dir is not None
    if (args.image_dir is None) != (args.mask_dir is None):
        parser.error("--image_dir and --mask_dir must be provided together")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, ckpt_cfg = load_model(args.checkpoint, device)
    data_cfg = cfg.get("data", {})
    image_size = data_cfg.get("image_size", 224)

    if custom_mode:
        print(f"\nCustom sequence evaluation:")
        print(f"  Images: {args.image_dir}")
        print(f"  Masks:  {args.mask_dir}")
        print(f"  Frame distance: {args.frame_distance}")

        test_dataset = SingleSequenceDataset(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            image_size=image_size,
            frame_distance=args.frame_distance,
        )
        label = "custom sequence"
    else:
        video_cfg = cfg.get("video", {})
        test_dataset = VideoSegDataset(
            dataset_root=video_cfg["dataset_root"],
            prompt_cache_path=video_cfg.get("prompt_cache", "./data/video_prompt_cache.json"),
            image_size=image_size,
            frame_distance_min=video_cfg.get("frame_distance_min", 3),
            frame_distance_max=video_cfg.get("frame_distance_max", 10),
            split="test",
            train_ratio=video_cfg.get("train_ratio", 0.8),
            val_ratio=video_cfg.get("val_ratio", 0.1),
            test_ratio=video_cfg.get("test_ratio", 0.1),
            seed=video_cfg.get("seed", data_cfg.get("seed", 42)),
        )
        test_sequences = [v["seq_name"] for v in test_dataset.videos]
        print(f"Test sequences ({len(test_sequences)}): {test_sequences}")
        label = f"{len(test_sequences)} test sequences"

    if len(test_dataset) == 0:
        print("No test pairs found.")
        return

    print(f"Test pairs: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.get("video", {}).get("batch_size", 4),
        shuffle=False,
        num_workers=cfg.get("training", {}).get("num_workers", 4),
        pin_memory=True,
        collate_fn=video_collate_fn,
    )

    loss_cfg = cfg["loss"]
    criterion = CombinedLoss(
        visual_dim=cfg["model"]["encoder_dim"],
        text_dim=cfg["model"]["text_dim"],
        lambda_1=loss_cfg.get("lambda_1", 1.0),
        lambda_2=loss_cfg.get("lambda_2", 1.0),
        lambda_3=loss_cfg.get("lambda_3", 0.5),
        lambda_4=loss_cfg.get("lambda_4", 1.0),
    ).to(device)

    metrics = evaluate_video_test(model, test_loader, criterion, device)
    _print_results(metrics, label, len(test_dataset))

    if args.output:
        results = {
            "overall": {k: float(v) for k, v in metrics.items()},
            "num_pairs": len(test_dataset),
        }
        if not custom_mode:
            results["test_sequences"] = [v["seq_name"] for v in test_dataset.videos]
        else:
            results["image_dir"] = args.image_dir
            results["mask_dir"] = args.mask_dir
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
