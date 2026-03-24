"""Waste Dataset with Ontario Category Support"""

from pathlib import Path
from typing import Optional, Callable, Tuple, Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

from ecosort.constants import CATEGORY_NAME_TO_ID


class WasteDataset(Dataset):
    """
    PyTorch Dataset for Ontario waste classification.

    Expected directory structure:
        data/processed/ontario/
        ├── train/
        │   ├── blue_bin/
        │   ├── green_bin/
        │   └── ...
        ├── val/
        └── test/
    """

    def __init__(
        self,
        root_dir: Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.split_dir = self.root_dir / split

        if not self.split_dir.exists():
            raise ValueError(f"Split directory does not exist: {self.split_dir}")

        # Build class mapping
        self.classes = sorted([d.name for d in self.split_dir.iterdir() if d.is_dir()])
        self.class_to_idx = class_to_idx or {
            cls: CATEGORY_NAME_TO_ID[cls] for cls in self.classes
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Build sample list
        self.samples = self._build_samples()

    def _build_samples(self) -> List[Tuple[Path, int]]:
        """Build list of (image_path, label) tuples."""
        samples = []
        for class_name in self.classes:
            class_dir = self.split_dir / class_name
            if not class_dir.is_dir():
                continue
            label = self.class_to_idx[class_name]
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    samples.append((img_path, label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        # Apply transforms
        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        """Return count of samples per class."""
        distribution = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            distribution[self.idx_to_class[label]] += 1
        return distribution


def create_dataloaders(
    data_dir: Path,
    transform_train: Callable,
    transform_val: Callable,
    batch_size: int = 8,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, test dataloaders."""

    train_dataset = WasteDataset(data_dir, split="train", transform=transform_train)
    val_dataset = WasteDataset(data_dir, split="val", transform=transform_val)
    test_dataset = WasteDataset(data_dir, split="test", transform=transform_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
