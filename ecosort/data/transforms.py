"""Albumentations Transforms for Training and Validation"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from ecosort.constants import IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.Affine(scale=(0.8, 1.2), rotate=(-30, 30), p=0.5),
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0), p=1.0),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(0.05, 0.15),
                hole_width_range=(0.05, 0.15),
                fill=0,
                p=0.3,
            ),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """
    Minimal transforms for validation/testing.
    Only resize, center crop, normalize.
    """
    return A.Compose(
        [
            A.Resize(height=256, width=256),
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_inference_transforms(image_size: int = 224) -> A.Compose:
    """
    Transforms for single image inference.
    Same as validation transforms.
    """
    return get_val_transforms(image_size)
