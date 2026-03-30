import json
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset

from .transforms import JointTransform


class KvasirSegDataset(Dataset):
    """Kvasir-SEG dataset with pre-computed text prompts."""

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
    import torch

    def collate_fn(batch):
        images, masks, prompts = zip(*batch)
        images = torch.stack(images)
        masks = torch.stack(masks)
        return images, masks, list(prompts)

    return collate_fn


def _split_indices(n, train_ratio, val_ratio, test_ratio, seed):
    import torch
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def get_dataloaders(cfg):
    """Create train, validation, and test dataloaders with 3-way split."""
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    dataset_root = data_cfg["dataset_root"]
    image_dir = os.path.join(dataset_root, "images")
    mask_dir = os.path.join(dataset_root, "masks")

    train_transform = JointTransform(data_cfg["image_size"], is_train=True)
    val_transform = JointTransform(data_cfg["image_size"], is_train=False)

    full_dataset = KvasirSegDataset(
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
