import argparse
import json
import os

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from data.dataset import get_dataloaders
from data.transforms import JointTransform
from models.segmentation_model import PolypSegmentationModel
from utils.metrics import compute_metrics


def load_model_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]

    model = PolypSegmentationModel(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    best_dice = checkpoint.get("best_dice", "N/A")
    print(f"Loaded checkpoint from epoch {epoch} (best Dice: {best_dice})")

    return model, cfg


@torch.no_grad()
def evaluate_loader(model, loader, device):
    """Evaluate model on a DataLoader (for built-in test split)."""
    model.eval()
    all_preds = []
    all_targets = []

    for images, masks, prompts in tqdm(loader, desc="Testing"):
        images = images.to(device)
        pred_mask, _, _, _ = model(images, prompts)
        pred_sigmoid = torch.sigmoid(pred_mask)

        all_preds.append(pred_sigmoid.cpu().numpy())
        all_targets.append(masks.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    overall = compute_metrics(all_preds, all_targets)

    return {
        "overall": {k: float(v) for k, v in overall.items()},
        "num_images": all_preds.shape[0],
    }


@torch.no_grad()
def evaluate_directory(model, image_dir, mask_dir, prompt_cache, image_size, device,
                       batch_size=8, default_prompt="polyp"):
    """Evaluate model on an external dataset directory."""
    transform = JointTransform(image_size=image_size, is_train=False)

    filenames = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    if not filenames:
        print(f"No images found in {image_dir}")
        return None

    print(f"Evaluating on {len(filenames)} images from {image_dir}")

    all_preds = []
    all_targets = []
    per_image_metrics = []

    for i in tqdm(range(0, len(filenames), batch_size), desc="Testing"):
        batch_fnames = filenames[i:i + batch_size]
        images = []
        masks = []
        prompts = []

        for fname in batch_fnames:
            image = Image.open(os.path.join(image_dir, fname)).convert("RGB")
            mask = Image.open(os.path.join(mask_dir, fname)).convert("L")
            prompt = prompt_cache.get(fname, default_prompt)

            image, mask = transform(image, mask)
            images.append(image)
            masks.append(mask)
            prompts.append(prompt)

        images = torch.stack(images).to(device)
        masks = torch.stack(masks).to(device)

        pred_mask, _, _, _ = model(images, prompts)
        pred_sigmoid = torch.sigmoid(pred_mask)

        pred_np = pred_sigmoid.cpu().numpy()
        mask_np = masks.cpu().numpy()

        all_preds.append(pred_np)
        all_targets.append(mask_np)

        for j in range(len(batch_fnames)):
            m = compute_metrics(pred_np[j:j+1], mask_np[j:j+1])
            per_image_metrics.append({
                "filename": batch_fnames[j],
                "dice": float(m["dice"]),
                "iou": float(m["iou"]),
                "f1": float(m["f1"]),
                "hausdorff": float(m["hausdorff"]),
            })

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    overall = compute_metrics(all_preds, all_targets)

    return {
        "overall": {k: float(v) for k, v in overall.items()},
        "per_image": per_image_metrics,
        "num_images": len(filenames),
    }


def print_results(results):
    print("\n" + "=" * 50)
    print(f"TEST RESULTS ({results['num_images']} images)")
    print("=" * 50)
    print(f"  Dice:              {results['overall']['dice']:.4f}")
    print(f"  IoU:               {results['overall']['iou']:.4f}")
    print(f"  F1:                {results['overall']['f1']:.4f}")
    print(f"  Hausdorff Dist:    {results['overall']['hausdorff']:.2f}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Test Polyp Segmentation Model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config (used for built-in test split)")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Path to external test images directory (optional)")
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="Path to external test masks directory (optional)")
    parser.add_argument("--prompt_cache", type=str, default=None,
                        help="Path to prompt_cache.json (optional)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON (optional)")
    parser.add_argument("--default_prompt", type=str, default="polyp",
                        help="Default prompt when no cache entry exists")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, ckpt_cfg = load_model_from_checkpoint(args.checkpoint, device)
    image_size = ckpt_cfg["data"]["image_size"]

    use_external = args.image_dir is not None and args.mask_dir is not None

    if use_external:
        prompt_cache = {}
        if args.prompt_cache and os.path.exists(args.prompt_cache):
            with open(args.prompt_cache) as f:
                prompt_cache = json.load(f)
            print(f"Loaded {len(prompt_cache)} prompts from {args.prompt_cache}")

        results = evaluate_directory(
            model=model,
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            prompt_cache=prompt_cache,
            image_size=image_size,
            device=device,
            batch_size=args.batch_size,
            default_prompt=args.default_prompt,
        )
    else:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        _, _, test_loader = get_dataloaders(cfg)

        if test_loader is None or len(test_loader.dataset) == 0:
            print("No test split configured (test_ratio is 0 or missing). "
                  "Use --image_dir and --mask_dir for external datasets.")
            return

        print(f"Using built-in test split ({len(test_loader.dataset)} images)")
        results = evaluate_loader(model, test_loader, device)

    if results is None:
        return

    print_results(results)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
