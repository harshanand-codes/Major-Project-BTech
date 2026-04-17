"""
Pre-compute text prompts for each mask based on polyp size and shape.

Run as a script:
    python -m data.prompt_generator --config configs/config.yaml

Categories (7 total):
    - "normal mucosa"              (empty mask)
    - "small round polyp"          (area < 5%, circularity > 0.7)
    - "small irregular polyp"      (area < 5%, circularity <= 0.7)
    - "medium round polyp"         (5% <= area < 20%, circularity > 0.7)
    - "medium irregular polyp"     (5% <= area < 20%, circularity <= 0.7)
    - "large round polyp"          (area >= 20%, circularity > 0.7)
    - "large irregular polyp"      (area >= 20%, circularity <= 0.7)
"""

import argparse
import json
import math
import os
from collections import Counter

import numpy as np
import yaml
from PIL import Image


class PromptGenerator:
    def __init__(self, size_small=0.05, size_large=0.20, circularity_thresh=0.7):
        self.size_small = size_small
        self.size_large = size_large
        self.circularity_thresh = circularity_thresh

    def compute_prompt(self, mask_path):
        mask = np.array(Image.open(mask_path).convert("L"))
        binary = (mask > 128).astype(np.uint8)

        area = binary.sum()
        total_pixels = binary.size

        if area == 0:
            return "normal mucosa"

        area_ratio = area / total_pixels
        size_label = self._get_size_label(area_ratio)

        circularity = self._compute_circularity(binary)
        shape_label = "round" if circularity > self.circularity_thresh else "irregular"

        return f"{size_label} {shape_label} polyp"

    def _get_size_label(self, area_ratio):
        if area_ratio < self.size_small:
            return "small"
        elif area_ratio < self.size_large:
            return "medium"
        else:
            return "large"

    def _compute_circularity(self, binary_mask):
        padded = np.pad(binary_mask, 1, mode="constant", constant_values=0)
        eroded = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
        boundary = binary_mask - eroded
        perimeter = boundary.sum()

        if perimeter == 0:
            return 1.0

        area = binary_mask.sum()
        circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
        return min(circularity, 1.0)

    def generate_all(self, mask_dir):
        """Generate prompts for a flat mask directory (image dataset)."""
        prompts = {}
        for fname in sorted(os.listdir(mask_dir)):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            prompts[fname] = self.compute_prompt(os.path.join(mask_dir, fname))
        return prompts

    def generate_all_video(self, dataset_root):
        """Generate prompts for a video dataset with <seq>/masks/<frame> layout."""
        prompts = {}
        for seq_name in sorted(os.listdir(dataset_root)):
            mask_dir = os.path.join(dataset_root, seq_name, "masks")
            if not os.path.isdir(mask_dir):
                continue
            for fname in sorted(os.listdir(mask_dir)):
                if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue
                prompts[f"{seq_name}/{fname}"] = self.compute_prompt(
                    os.path.join(mask_dir, fname)
                )
        return prompts


def _save_and_report(prompts, output_path, label):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(prompts, f, indent=2)

    print(f"\n[{label}] Saved {len(prompts)} prompts to {output_path}")
    counts = Counter(prompts.values())
    for prompt, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {prompt}: {count} ({100 * count / len(prompts):.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute prompts for all datasets defined in the config"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    prompt_cfg = cfg.get("prompt", {})
    thresholds = prompt_cfg.get("size_thresholds", {})
    generator = PromptGenerator(
        size_small=thresholds.get("small", 0.05),
        size_large=thresholds.get("large", 0.20),
        circularity_thresh=prompt_cfg.get("circularity_threshold", 0.7),
    )

    data_cfg = cfg.get("data", {})
    if "dataset_root" in data_cfg:
        mask_dir = os.path.join(data_cfg["dataset_root"], "masks")
        output = data_cfg.get("prompt_cache", "./data/prompt_cache.json")
        print(f"Scanning image masks in: {mask_dir}")
        prompts = generator.generate_all(mask_dir)
        _save_and_report(prompts, output, "image")

    video_cfg = cfg.get("video", {})
    if "dataset_root" in video_cfg:
        video_root = video_cfg["dataset_root"]
        output = video_cfg.get("prompt_cache", "./data/video_prompt_cache.json")
        print(f"\nScanning video masks in: {video_root}")
        prompts = generator.generate_all_video(video_root)
        _save_and_report(prompts, output, "video")


if __name__ == "__main__":
    main()
