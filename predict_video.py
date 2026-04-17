"""
Video inference: runs the full pipeline (STEP 1 + STEP 2 + STEP 3) on a pair
of frames, using cross-frame correspondence to enhance segmentation.
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from models.segmentation_model import PolypSegmentationModel


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


def make_transform(image_size):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def make_overlay(original, mask_img):
    orig_w, orig_h = original.size
    overlay = original.copy().convert("RGBA")
    green = Image.new("RGBA", (orig_w, orig_h), (0, 255, 0, 0))
    mask_rgba = mask_img.point(lambda p: 100 if p > 0 else 0)
    green.putalpha(mask_rgba)
    return Image.alpha_composite(overlay, green).convert("RGB")


@torch.no_grad()
def predict_pair(model, image_path1, image_path2, image_size, device,
                 prompt="polyp", threshold=0.5):
    """
    Run inference using two frames with cross-frame correspondence.
    Segmentation is produced for frame 1, enhanced by correspondence with frame 2.
    """
    original1 = Image.open(image_path1).convert("RGB")
    original2 = Image.open(image_path2).convert("RGB")
    orig_w, orig_h = original1.size

    transform = make_transform(image_size)
    t1 = transform(original1).unsqueeze(0).to(device)
    t2 = transform(original2).unsqueeze(0).to(device)

    pred_mask, _, _, _ = model(t1, [prompt], images2=t2)
    pred_prob = torch.sigmoid(pred_mask).squeeze().cpu().numpy()

    pred_bin = (pred_prob > threshold).astype(np.uint8) * 255
    mask_img = Image.fromarray(pred_bin, mode="L")
    mask_img = mask_img.resize((orig_w, orig_h), Image.NEAREST)

    overlay = make_overlay(original1, mask_img)

    return mask_img, overlay


@torch.no_grad()
def predict_video_folder(model, frames_dir, image_size, device,
                         prompt="polyp", threshold=0.5, frame_distance=5):
    """
    Run inference on all frames in a video folder using adjacent frame pairs.
    Each frame is segmented using correspondence with a neighboring frame.
    """
    frames = sorted(
        [f for f in os.listdir(frames_dir)
         if f.lower().endswith((".jpg", ".png", ".jpeg"))],
        key=lambda fn: int(os.path.splitext(fn)[0]),
    )

    if not frames:
        print(f"No frames found in {frames_dir}")
        return []

    transform = make_transform(image_size)
    results = []

    for i, fname in enumerate(frames):
        neighbor_idx = min(i + frame_distance, len(frames) - 1)
        if neighbor_idx == i:
            neighbor_idx = max(i - frame_distance, 0)

        original = Image.open(os.path.join(frames_dir, fname)).convert("RGB")
        neighbor = Image.open(os.path.join(frames_dir, frames[neighbor_idx])).convert("RGB")
        orig_w, orig_h = original.size

        t1 = transform(original).unsqueeze(0).to(device)
        t2 = transform(neighbor).unsqueeze(0).to(device)

        pred_mask, _, _, _ = model(t1, [prompt], images2=t2)
        pred_prob = torch.sigmoid(pred_mask).squeeze().cpu().numpy()

        pred_bin = (pred_prob > threshold).astype(np.uint8) * 255
        mask_img = Image.fromarray(pred_bin, mode="L")
        mask_img = mask_img.resize((orig_w, orig_h), Image.NEAREST)

        overlay = make_overlay(original, mask_img)

        results.append({
            "filename": fname,
            "neighbor": frames[neighbor_idx],
            "mask": mask_img,
            "overlay": overlay,
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Video inference with cross-frame correspondence"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pair", nargs=2, metavar=("FRAME1", "FRAME2"),
                       help="Two frame paths for pair inference")
    group.add_argument("--video_dir", type=str,
                       help="Directory of video frames for full video inference")

    parser.add_argument("--output_dir", type=str, default="./predictions_video",
                        help="Directory to save output images")
    parser.add_argument("--prompt", type=str, default="polyp")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--frame_distance", type=int, default=5,
                        help="Frame distance for neighbor pairing (video_dir mode)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, image_size = load_model(args.checkpoint, device)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.pair:
        frame1_path, frame2_path = args.pair
        print(f"Pair inference: {frame1_path} + {frame2_path}")

        mask_img, overlay = predict_pair(
            model, frame1_path, frame2_path,
            image_size, device, args.prompt, args.threshold
        )

        basename = os.path.splitext(os.path.basename(frame1_path))[0]
        mask_path = os.path.join(args.output_dir, f"{basename}_mask.png")
        overlay_path = os.path.join(args.output_dir, f"{basename}_overlay.png")

        mask_img.save(mask_path)
        overlay.save(overlay_path)
        print(f"Mask saved to:    {mask_path}")
        print(f"Overlay saved to: {overlay_path}")

    else:
        print(f"Video inference: {args.video_dir} (frame_distance={args.frame_distance})")

        results = predict_video_folder(
            model, args.video_dir, image_size, device,
            args.prompt, args.threshold, args.frame_distance
        )

        for r in results:
            basename = os.path.splitext(r["filename"])[0]
            r["mask"].save(os.path.join(args.output_dir, f"{basename}_mask.png"))
            r["overlay"].save(os.path.join(args.output_dir, f"{basename}_overlay.png"))

        print(f"Saved {len(results)} masks and overlays to {args.output_dir}")


if __name__ == "__main__":
    main()
