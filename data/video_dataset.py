import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class LDPolypVideoDataset(Dataset):
    """
    LDPolypVideo frame-pair dataset.

    Each sample returns two frames from the same video (3-10 frames apart)
    and their bounding box annotations.
    """

    def __init__(self, dataset_root, image_size=224,
                 frame_distance_min=3, frame_distance_max=10,
                 samples_per_epoch=1000):
        self.image_size = image_size
        self.frame_distance_min = frame_distance_min
        self.frame_distance_max = frame_distance_max
        self.samples_per_epoch = samples_per_epoch

        images_root = os.path.join(dataset_root, "Images")
        annotations_root = os.path.join(dataset_root, "Annotations")

        self.videos = []
        for vid_name in sorted(os.listdir(images_root)):
            vid_img_dir = os.path.join(images_root, vid_name)
            vid_ann_dir = os.path.join(annotations_root, vid_name)
            if not os.path.isdir(vid_img_dir):
                continue

            frames = sorted([
                f for f in os.listdir(vid_img_dir)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ])
            if len(frames) < frame_distance_min + 1:
                continue

            self.videos.append({
                "img_dir": vid_img_dir,
                "ann_dir": vid_ann_dir,
                "frames": frames,
            })

        self.image_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return self.samples_per_epoch

    def _parse_bboxes(self, ann_path, orig_w, orig_h):
        """Parse bounding boxes from annotation file, normalize to [0, 1]."""
        if not os.path.exists(ann_path):
            return torch.zeros(0, 4)

        with open(ann_path, "r") as f:
            lines = f.read().strip().split("\n")

        num_boxes = int(lines[0])
        boxes = []
        for i in range(1, min(num_boxes + 1, len(lines))):
            parts = lines[i].strip().split()
            if len(parts) >= 4:
                x1, y1, x2, y2 = float(parts[0]), float(parts[1]), \
                                  float(parts[2]), float(parts[3])
                boxes.append([
                    x1 / orig_w, y1 / orig_h,
                    x2 / orig_w, y2 / orig_h,
                ])

        if not boxes:
            return torch.zeros(0, 4)
        return torch.tensor(boxes, dtype=torch.float32)

    def __getitem__(self, idx):
        video = random.choice(self.videos)
        frames = video["frames"]
        max_idx = len(frames) - 1

        idx1 = random.randint(0, max_idx)
        distance = random.randint(self.frame_distance_min, self.frame_distance_max)
        idx2 = min(idx1 + distance, max_idx)
        if idx2 == idx1:
            idx2 = min(idx1 + 1, max_idx)

        img1 = Image.open(os.path.join(video["img_dir"], frames[idx1])).convert("RGB")
        img2 = Image.open(os.path.join(video["img_dir"], frames[idx2])).convert("RGB")
        orig_w, orig_h = img1.size

        ann_name1 = os.path.splitext(frames[idx1])[0] + ".txt"
        ann_name2 = os.path.splitext(frames[idx2])[0] + ".txt"
        bbox1 = self._parse_bboxes(
            os.path.join(video["ann_dir"], ann_name1), orig_w, orig_h
        )
        bbox2 = self._parse_bboxes(
            os.path.join(video["ann_dir"], ann_name2), orig_w, orig_h
        )

        img1 = self.image_transform(img1)
        img2 = self.image_transform(img2)

        return img1, img2, bbox1, bbox2


def video_collate_fn(batch):
    """Collate for variable-count bounding boxes."""
    imgs1, imgs2, bboxes1, bboxes2 = zip(*batch)
    imgs1 = torch.stack(imgs1)
    imgs2 = torch.stack(imgs2)
    return imgs1, imgs2, list(bboxes1), list(bboxes2)
