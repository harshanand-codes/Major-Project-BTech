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


@torch.no_grad()
def predict(model, image_path, image_size, device, prompt="polyp", threshold=0.5):
    original = Image.open(image_path).convert("RGB")
    orig_w, orig_h = original.size

    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = transform(original).unsqueeze(0).to(device)

    pred_mask, _, _, _ = model(input_tensor, [prompt])
    pred_prob = torch.sigmoid(pred_mask).squeeze().cpu().numpy()

    pred_bin = (pred_prob > threshold).astype(np.uint8) * 255
    mask_img = Image.fromarray(pred_bin, mode="L")
    mask_img = mask_img.resize((orig_w, orig_h), Image.NEAREST)

    overlay = original.copy().convert("RGBA")
    green = Image.new("RGBA", (orig_w, orig_h), (0, 255, 0, 0))
    mask_rgba = mask_img.point(lambda p: 100 if p > 0 else 0)
    green.putalpha(mask_rgba)
    overlay = Image.alpha_composite(overlay, green).convert("RGB")

    return mask_img, overlay, pred_prob


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="./predictions",
                        help="Directory to save output images")
    parser.add_argument("--prompt", type=str, default="polyp",
                        help="Text prompt for the model")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Binarization threshold for the mask")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, image_size = load_model(args.checkpoint, device)

    mask_img, overlay, prob_map = predict(
        model, args.image, image_size, device, args.prompt, args.threshold
    )

    os.makedirs(args.output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.image))[0]

    mask_path = os.path.join(args.output_dir, f"{basename}_mask.png")
    overlay_path = os.path.join(args.output_dir, f"{basename}_overlay.png")

    mask_img.save(mask_path)
    overlay.save(overlay_path)

    print(f"Mask saved to:    {mask_path}")
    print(f"Overlay saved to: {overlay_path}")


if __name__ == "__main__":
    main()
