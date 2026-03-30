"""
Pre-compute text prompts for each mask in Kvasir-SEG based on polyp size and shape.

Run as a script:
    python -m data.prompt_generator --mask_dir ./Kvasir-SEG/masks --output ./data/prompt_cache.json

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
        # Perimeter: count boundary pixels (pixels with at least one 0-neighbor)
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
        prompts = {}
        for fname in sorted(os.listdir(mask_dir)):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            path = os.path.join(mask_dir, fname)
            prompts[fname] = self.compute_prompt(path)
        return prompts


def main():
    parser = argparse.ArgumentParser(description="Pre-compute prompts for Kvasir-SEG")
    parser.add_argument("--mask_dir", type=str, default="./Kvasir-SEG/masks")
    parser.add_argument("--output", type=str, default="./data/prompt_cache.json")
    parser.add_argument("--size_small", type=float, default=0.05)
    parser.add_argument("--size_large", type=float, default=0.20)
    parser.add_argument("--circularity_thresh", type=float, default=0.7)
    args = parser.parse_args()

    generator = PromptGenerator(
        size_small=args.size_small,
        size_large=args.size_large,
        circularity_thresh=args.circularity_thresh,
    )

    print(f"Scanning masks in: {args.mask_dir}")
    prompts = generator.generate_all(args.mask_dir)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(prompts, f, indent=2)

    print(f"\nSaved {len(prompts)} prompts to {args.output}")
    print("\nPrompt distribution:")
    counts = Counter(prompts.values())
    for prompt, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {prompt}: {count} ({100*count/len(prompts):.1f}%)")


if __name__ == "__main__":
    main()
