"""
Video test: evaluates segmentation on video frame pairs using the full pipeline
(STEP 1 + STEP 2 + STEP 3). Requires a video dataset with frame images and
ground truth segmentation masks.
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from models.segmentation_model import PolypSegmentationModel
from utils.metrics import compute_metrics


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]

    model = PolypSegmentationModel(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_size = cfg["data"]["image_size"]
    print(f"Loaded checkpoint (epoch {checkpoint.get('epoch', '?')}, "
          f"Dice {checkpoint.get('best_dice', 'N/A')})")

    return model, image_size


@torch.no_grad()
def evaluate_video_dataset(model, image_root, mask_root, image_size, device,
                           frame_distance=5, threshold=0.5, prompt="polyp"):
    """
    Evaluate on a video dataset with structure:
        image_root/<video_id>/<frame>.jpg
        mask_root/<video_id>/<frame>.png (binary masks)

    Each frame is segmented using correspondence with a neighbor frame.
    """
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_preds = []
    all_targets = []
    per_video_metrics = []

    video_ids = sorted([
        d for d in os.listdir(image_root)
        if os.path.isdir(os.path.join(image_root, d))
    ])

    if not video_ids:
        print(f"No video directories found in {image_root}")
        return None

    total_frames = 0

    for vid_id in tqdm(video_ids, desc="Videos"):
        vid_img_dir = os.path.join(image_root, vid_id)
        vid_mask_dir = os.path.join(mask_root, vid_id)

        if not os.path.isdir(vid_mask_dir):
            continue

        frames = sorted([
            f for f in os.listdir(vid_img_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        mask_files = set(os.listdir(vid_mask_dir))
        vid_preds = []
        vid_targets = []

        for i, fname in enumerate(frames):
            mask_name_png = os.path.splitext(fname)[0] + ".png"
            mask_name_jpg = os.path.splitext(fname)[0] + ".jpg"

            if mask_name_png in mask_files:
                mask_fname = mask_name_png
            elif mask_name_jpg in mask_files:
                mask_fname = mask_name_jpg
            else:
                continue

            neighbor_idx = min(i + frame_distance, len(frames) - 1)
            if neighbor_idx == i:
                neighbor_idx = max(i - frame_distance, 0)

            img = Image.open(os.path.join(vid_img_dir, fname)).convert("RGB")
            neighbor = Image.open(
                os.path.join(vid_img_dir, frames[neighbor_idx])
            ).convert("RGB")

            t1 = transform(img).unsqueeze(0).to(device)
            t2 = transform(neighbor).unsqueeze(0).to(device)

            pred_mask, _, _, _ = model(t1, [prompt], images2=t2)
            pred_prob = torch.sigmoid(pred_mask).squeeze().cpu().numpy()

            mask = Image.open(os.path.join(vid_mask_dir, mask_fname)).convert("L")
            mask = mask.resize((image_size, image_size), Image.NEAREST)
            mask_np = (np.array(mask) > 128).astype(np.float32)

            vid_preds.append(pred_prob[np.newaxis, np.newaxis, ...])
            vid_targets.append(mask_np[np.newaxis, np.newaxis, ...])
            total_frames += 1

        if vid_preds:
            vp = np.concatenate(vid_preds, axis=0)
            vt = np.concatenate(vid_targets, axis=0)
            all_preds.append(vp)
            all_targets.append(vt)

            vm = compute_metrics(vp, vt, threshold=threshold)
            per_video_metrics.append({
                "video_id": vid_id,
                "num_frames": len(vid_preds),
                "dice": float(vm["dice"]),
                "iou": float(vm["iou"]),
                "f1": float(vm["f1"]),
                "hausdorff": float(vm["hausdorff"]),
            })

    if not all_preds:
        print("No frames with masks found.")
        return None

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    overall = compute_metrics(all_preds, all_targets, threshold=threshold)

    return {
        "overall": {k: float(v) for k, v in overall.items()},
        "per_video": per_video_metrics,
        "num_frames": total_frames,
        "num_videos": len(per_video_metrics),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test segmentation on video dataset with cross-frame correspondence"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--image_root", type=str, required=True,
                        help="Root directory of video frames (image_root/<vid>/<frame>.jpg)")
    parser.add_argument("--mask_root", type=str, required=True,
                        help="Root directory of video masks (mask_root/<vid>/<frame>.png)")
    parser.add_argument("--frame_distance", type=int, default=5,
                        help="Frame distance for neighbor pairing")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--prompt", type=str, default="polyp")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, image_size = load_model(args.checkpoint, device)

    results = evaluate_video_dataset(
        model=model,
        image_root=args.image_root,
        mask_root=args.mask_root,
        image_size=image_size,
        device=device,
        frame_distance=args.frame_distance,
        threshold=args.threshold,
        prompt=args.prompt,
    )

    if results is None:
        return

    print("\n" + "=" * 50)
    print(f"VIDEO TEST RESULTS ({results['num_frames']} frames, "
          f"{results['num_videos']} videos)")
    print("=" * 50)
    print(f"  Dice:              {results['overall']['dice']:.4f}")
    print(f"  IoU:               {results['overall']['iou']:.4f}")
    print(f"  F1:                {results['overall']['f1']:.4f}")
    print(f"  Hausdorff Dist:    {results['overall']['hausdorff']:.2f}")
    print("=" * 50)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
