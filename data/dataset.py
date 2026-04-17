import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from .transforms import JointTransform
from .video_dataset import VideoSegDataset, video_collate_fn


class ImageSegDataset(Dataset):
    """Image segmentation dataset with pre-computed text prompts."""

    def __init__(self, image_dir, mask_dir, prompt_cache_path, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        with open(prompt_cache_path, "r") as f:
            self.prompt_cache = json.load(f)

        self.filenames = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        image = Image.open(os.path.join(self.image_dir, fname)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, fname)).convert("L")
        prompt = self.prompt_cache.get(fname, "polyp")

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask, prompt


def _make_collate_fn():
    def collate_fn(batch):
        images, masks, prompts = zip(*batch)
        images = torch.stack(images)
        masks = torch.stack(masks)
        return images, masks, list(prompts)

    return collate_fn


def _split_indices(n, train_ratio, val_ratio, test_ratio, seed):
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


class _SubsetWithTransform(Dataset):
    """Wraps a dataset subset with a specific transform."""

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        fname = self.dataset.filenames[real_idx]

        image = Image.open(
            os.path.join(self.dataset.image_dir, fname)
        ).convert("RGB")
        mask = Image.open(
            os.path.join(self.dataset.mask_dir, fname)
        ).convert("L")
        prompt = self.dataset.prompt_cache.get(fname, "polyp")

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask, prompt


def get_dataloaders(cfg):
    """Create train, validation, and test dataloaders (image segmentation only)."""
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    dataset_root = data_cfg["dataset_root"]
    image_dir = os.path.join(dataset_root, "images")
    mask_dir = os.path.join(dataset_root, "masks")

    train_transform = JointTransform(data_cfg["image_size"], is_train=True)
    val_transform = JointTransform(data_cfg["image_size"], is_train=False)

    full_dataset = ImageSegDataset(
        image_dir, mask_dir, data_cfg["prompt_cache"], transform=None
    )

    train_ratio = data_cfg["train_ratio"]
    val_ratio = data_cfg.get("val_ratio", 1.0 - train_ratio)
    test_ratio = data_cfg.get("test_ratio", 0.0)

    train_indices, val_indices, test_indices = _split_indices(
        len(full_dataset), train_ratio, val_ratio, test_ratio, data_cfg["seed"]
    )

    train_dataset = _SubsetWithTransform(full_dataset, train_indices, train_transform)
    val_dataset = _SubsetWithTransform(full_dataset, val_indices, val_transform)

    collate_fn = _make_collate_fn()

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_loader = None
    if test_indices:
        test_dataset = _SubsetWithTransform(full_dataset, test_indices, val_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            num_workers=train_cfg["num_workers"],
            pin_memory=True,
            collate_fn=collate_fn,
        )

    return train_loader, val_loader, test_loader


def get_mixed_dataloaders(cfg):
    """
    Create mixed training dataloaders for joint image segmentation + video segmentation training.

    Returns:
        seg_train_loader, video_train_loader, val_loader, test_loader,
        video_val_loader, video_test_loader
    """
    seg_train_loader, val_loader, test_loader = get_dataloaders(cfg)

    video_cfg = cfg.get("video", {})
    video_root = video_cfg.get("dataset_root", "/root/dataset_collection/polypgen/positive_cropped")
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    video_batch_size = video_cfg.get("batch_size", train_cfg["batch_size"] // 2)
    prompt_cache = video_cfg.get("prompt_cache", "./data/video_prompt_cache.json")
    dist_min = video_cfg.get("frame_distance_min", 3)
    dist_max = video_cfg.get("frame_distance_max", 10)
    v_train_ratio = video_cfg.get("train_ratio", 0.8)
    v_val_ratio = video_cfg.get("val_ratio", 0.1)
    v_test_ratio = video_cfg.get("test_ratio", 0.1)
    v_seed = video_cfg.get("seed", data_cfg.get("seed", 42))

    shared_kwargs = dict(
        dataset_root=video_root,
        prompt_cache_path=prompt_cache,
        image_size=data_cfg["image_size"],
        frame_distance_min=dist_min,
        frame_distance_max=dist_max,
        train_ratio=v_train_ratio,
        val_ratio=v_val_ratio,
        test_ratio=v_test_ratio,
        seed=v_seed,
    )

    video_train_dataset = VideoSegDataset(split="train", **shared_kwargs)
    video_val_dataset = VideoSegDataset(split="val", **shared_kwargs)
    video_test_dataset = VideoSegDataset(split="test", **shared_kwargs)

    video_train_loader = DataLoader(
        video_train_dataset,
        batch_size=video_batch_size,
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        collate_fn=video_collate_fn,
    )
    video_val_loader = DataLoader(
        video_val_dataset,
        batch_size=video_batch_size,
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        collate_fn=video_collate_fn,
    )
    video_test_loader = DataLoader(
        video_test_dataset,
        batch_size=video_batch_size,
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        collate_fn=video_collate_fn,
    ) if len(video_test_dataset) > 0 else None

    return (seg_train_loader, video_train_loader, val_loader, test_loader,
            video_val_loader, video_test_loader)
