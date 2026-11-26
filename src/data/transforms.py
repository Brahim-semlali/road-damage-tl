"""Preprocessing and data augmentation (Albumentations).

* Resizing: every image is resized to a **square** ``image_size`` (224). Images of the dataset
  come in different resolutions/aspect ratios depending on their source (see EDA); squashing
  them all to the same square size removes resolution as a trivial cue.
* Normalisation: ImageNet mean/std, expected by the pretrained backbones (also used for the
  scratch CNN so that all models see exactly the same inputs).
* Augmentation (train only): geometric (flip, small rotation/scale/shift, random crop) and
  photometric (brightness, contrast, colour, blur) variations that keep the road state unchanged.
  Validation and test images are only resized and normalised (no augmentation, no leakage).
"""
from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transforms(image_size: int = 224) -> A.Compose:
    resize = int(image_size * 1.14)  # 224 -> 255, then random 224 crop
    return A.Compose([
        A.Resize(resize, resize),
        A.RandomCrop(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-10, 10), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def eval_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
