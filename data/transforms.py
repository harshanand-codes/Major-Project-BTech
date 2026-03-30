import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as TF


class JointTransform:
    """Applies synchronized transforms to image and mask."""

    def __init__(self, image_size=224, is_train=True):
        self.image_size = image_size
        self.is_train = is_train

        self.image_normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.color_jitter = T.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        )

    def __call__(self, image, mask):
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        if self.is_train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            angle = random.choice([0, 90, 180, 270])
            if angle > 0:
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

            image = self.color_jitter(image)

        image = TF.to_tensor(image)
        image = self.image_normalize(image)

        mask = np.array(mask.convert("L"))
        mask = (mask > 128).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask
