"""TrashNet to Ontario Category Mapping"""

from pathlib import Path
from typing import Dict
import shutil

from ecosort.constants import TRASHNET_TO_ONTARIO


def map_trashnet_to_ontario(
    trashnet_dir: Path, output_dir: Path, split_ratios: Dict[str, float] = None
):
    """Map TrashNet dataset to Ontario categories."""
    if split_ratios is None:
        split_ratios = {"train": 0.7, "val": 0.15, "test": 0.15}

    # Create output directories
    for split in split_ratios:
        for ontario_class in set(TRASHNET_TO_ONTARIO.values()):
            (output_dir / split / ontario_class).mkdir(parents=True, exist_ok=True)

    # Map and copy images
    for trashnet_class, ontario_class in TRASHNET_TO_ONTARIO.items():
        src_class_dir = trashnet_dir / trashnet_class
        if not src_class_dir.exists():
            print(f"Warning: {src_class_dir} not found, skipping")
            continue

        images = list(src_class_dir.glob("*.jpg")) + list(src_class_dir.glob("*.png"))
        n = len(images)

        if n == 0:
            continue

        # Simple split
        train_end = int(n * split_ratios["train"])
        val_end = train_end + int(n * split_ratios["val"])

        for i, img_path in enumerate(images):
            if i < train_end:
                split = "train"
            elif i < val_end:
                split = "val"
            else:
                split = "test"

            dst = output_dir / split / ontario_class / img_path.name
            shutil.copy2(img_path, dst)

    print(f"Mapped TrashNet to Ontario categories in {output_dir}")


def get_ontario_class_for_trashnet(trashnet_class: str) -> str:
    """Get Ontario category for a TrashNet class."""
    return TRASHNET_TO_ONTARIO.get(trashnet_class, "garbage")
