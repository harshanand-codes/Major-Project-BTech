import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF


def _numeric_sort_key(filename):
    """Sort by integer stem so that 2.jpg comes before 10.jpg."""
    return int(os.path.splitext(filename)[0])


class VideoSegDataset(Dataset):
    """
    Video segmentation frame-pair dataset for PolypGen-style layouts.

    Expected directory structure:
        dataset_root/<seq>/images/<frame>.jpg
        dataset_root/<seq>/masks/<frame>.jpg

    Each sample returns two frames from the same sequence (a configurable
    number of frames apart) together with their binary segmentation masks.
    """

    def __init__(self, dataset_root, prompt_cache_path=None, image_size=224,
                 frame_distance_min=3, frame_distance_max=10,
                 split="train", train_ratio=0.8, val_ratio=0.1,
                 test_ratio=0.1, seed=42):
        self.image_size = image_size
        self.is_train = (split == "train")

        if prompt_cache_path and os.path.exists(prompt_cache_path):
            with open(prompt_cache_path, "r") as f:
                self.prompt_cache = json.load(f)
        else:
            self.prompt_cache = {}

        all_videos = []
        for seq_name in sorted(os.listdir(dataset_root)):
            seq_dir = os.path.join(dataset_root, seq_name)
            if not os.path.isdir(seq_dir):
                continue

            img_dir = os.path.join(seq_dir, "images")
            mask_dir = os.path.join(seq_dir, "masks")
            if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
                continue

            frames = sorted(
                [f for f in os.listdir(img_dir)
                 if f.lower().endswith((".jpg", ".png", ".jpeg"))],
                key=_numeric_sort_key,
            )
            if len(frames) < frame_distance_min + 1:
                continue

            all_videos.append({
                "seq_name": seq_name,
                "img_dir": img_dir,
                "mask_dir": mask_dir,
                "frames": frames,
            })

        n = len(all_videos)
        indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        if split == "train":
            keep = indices[:train_end]
        elif split == "val":
            keep = indices[train_end:val_end]
        else:
            keep = indices[val_end:]

        self.videos = [all_videos[i] for i in keep]

        self.pairs = []
        for vid_idx, video in enumerate(self.videos):
            n_frames = len(video["frames"])
            for i in range(n_frames):
                for d in range(frame_distance_min, frame_distance_max + 1):
                    j = i + d
                    if j < n_frames:
                        self.pairs.append((vid_idx, i, j))

        self.image_normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.pairs)

    def _apply_joint_augmentation(self, img1, img2, mask1, mask2):
        """Apply identical geometric augmentations to both frame-mask pairs."""
        if random.random() > 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
            mask1 = TF.hflip(mask1)
            mask2 = TF.hflip(mask2)

        if random.random() > 0.5:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)
            mask1 = TF.vflip(mask1)
            mask2 = TF.vflip(mask2)

        angle = random.choice([0, 90, 180, 270])
        if angle > 0:
            img1 = TF.rotate(img1, angle)
            img2 = TF.rotate(img2, angle)
            mask1 = TF.rotate(mask1, angle)
            mask2 = TF.rotate(mask2, angle)

        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = (
            T.ColorJitter.get_params(
                brightness=(0.8, 1.2),
                contrast=(0.8, 1.2),
                saturation=(0.8, 1.2),
                hue=(-0.05, 0.05),
            )
        )
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img1 = TF.adjust_brightness(img1, brightness_factor)
                img2 = TF.adjust_brightness(img2, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img1 = TF.adjust_contrast(img1, contrast_factor)
                img2 = TF.adjust_contrast(img2, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img1 = TF.adjust_saturation(img1, saturation_factor)
                img2 = TF.adjust_saturation(img2, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img1 = TF.adjust_hue(img1, hue_factor)
                img2 = TF.adjust_hue(img2, hue_factor)

        return img1, img2, mask1, mask2

    def _process_mask(self, mask_pil):
        """Resize, binarize, and convert a mask PIL image to a tensor."""
        mask_pil = mask_pil.resize(
            (self.image_size, self.image_size), Image.NEAREST
        )
        mask_np = np.array(mask_pil.convert("L"))
        mask_np = (mask_np > 128).astype(np.float32)
        return torch.from_numpy(mask_np).unsqueeze(0)

    def __getitem__(self, idx):
        vid_idx, idx1, idx2 = self.pairs[idx]
        video = self.videos[vid_idx]
        frames = video["frames"]

        img1 = Image.open(os.path.join(video["img_dir"], frames[idx1])).convert("RGB")
        img2 = Image.open(os.path.join(video["img_dir"], frames[idx2])).convert("RGB")
        mask1 = Image.open(os.path.join(video["mask_dir"], frames[idx1])).convert("L")
        mask2 = Image.open(os.path.join(video["mask_dir"], frames[idx2])).convert("L")

        img1 = img1.resize((self.image_size, self.image_size), Image.BILINEAR)
        img2 = img2.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask1 = mask1.resize((self.image_size, self.image_size), Image.NEAREST)
        mask2 = mask2.resize((self.image_size, self.image_size), Image.NEAREST)

        if self.is_train:
            img1, img2, mask1, mask2 = self._apply_joint_augmentation(
                img1, img2, mask1, mask2
            )

        prompt1 = self.prompt_cache.get(
            f"{video['seq_name']}/{frames[idx1]}", "polyp"
        )

        img1 = self.image_normalize(TF.to_tensor(img1))
        img2 = self.image_normalize(TF.to_tensor(img2))
        mask1 = self._process_mask(mask1)
        mask2 = self._process_mask(mask2)

        return img1, img2, mask1, mask2, prompt1


def video_collate_fn(batch):
    """Collate video frame pairs with their segmentation masks and prompts."""
    imgs1, imgs2, masks1, masks2, prompts = zip(*batch)
    return torch.stack(imgs1), torch.stack(imgs2), torch.stack(masks1), torch.stack(masks2), list(prompts)
