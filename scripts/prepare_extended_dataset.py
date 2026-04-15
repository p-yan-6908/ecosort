#!/usr/bin/env python3
"""
Prepare extended dataset for EcoSort training.

This script merges multiple waste datasets and maps them to Ontario's
6 waste categories:
- Blue Bin (Recyclables)
- Green Bin (Organics)  
- Garbage
- Household Hazardous
- E-Waste
- Yard Waste

Usage:
    python scripts/prepare_extended_dataset.py --data-dir data

The script will:
1. Scan all downloaded datasets
2. Map categories to Ontario standards
3. Split into train/val/test (70/15/15)
4. Balance classes to ensure good coverage
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import json
import logging

from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ontario category definitions
ONTARIO_CATEGORIES = {
    "blue_bin": "Blue Bin (Recyclables)",
    "green_bin": "Green Bin (Organics)",
    "garbage": "Garbage",
    "hazardous": "Household Hazardous",
    "e_waste": "Electronic Waste",
    "yard_waste": "Yard Waste",
}

# Category mappings from various datasets to Ontario categories
CATEGORY_MAPPINGS = {
    # TrashNet categories
    "glass": "blue_bin",
    "paper": "blue_bin",
    "cardboard": "blue_bin",
    "plastic": "blue_bin",
    "metal": "blue_bin",
    "trash": "garbage",
    
    # RealWaste categories
    "Cardboard": "blue_bin",
    "Food Organics": "green_bin",
    "Glass": "blue_bin",
    "Metal": "blue_bin",
    "Miscellaneous Trash": "garbage",
    "Paper": "blue_bin",
    "Plastic": "blue_bin",
    "Textile Trash": "garbage",
    "Vegetation": "yard_waste",
    
    # Kaggle Garbage Classification
    "battery": "hazardous",
    "biological": "green_bin",
    "brown-glass": "blue_bin",
    "cardboard": "blue_bin",
    "clothes": "garbage",
    "green-glass": "blue_bin",
    "metal": "blue_bin",
    "paper": "blue_bin",
    "plastic": "blue_bin",
    "shoes": "garbage",
    "trash": "garbage",
    "white-glass": "blue_bin",
    
    # E-Waste categories
    "laptop": "e_waste",
    "mobile": "e_waste",
    "monitor": "e_waste",
    "keyboard": "e_waste",
    "mouse": "e_waste",
    "cable": "e_waste",
    "battery": "hazardous",
    
    # Hazardous categories
    "battery": "hazardous",
    "paint": "hazardous",
    "chemical": "hazardous",
    "aerosol": "hazardous",
    
    # Organic/Yard categories
    "food": "green_bin",
    "compost": "green_bin",
    "leaves": "yard_waste",
    "grass": "yard_waste",
    "branches": "yard_waste",
    "yard": "yard_waste",
}

# Extensions to process
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}


def scan_dataset_directory(dataset_path: Path) -> Dict[str, List[Path]]:
    """Scan a dataset directory for images organized by category."""
    images_by_category = {}
    
    for category_dir in dataset_path.iterdir():
        if not category_dir.is_dir():
            continue
        
        category_name = category_dir.name.lower()
        images = []
        
        for ext in IMAGE_EXTENSIONS:
            images.extend(category_dir.glob(f"*{ext}"))
            images.extend(category_dir.glob(f"*{ext.upper()}"))
        
        if images:
            images_by_category[category_name] = images
    
    return images_by_category


def map_to_ontario_category(source_category: str) -> str:
    """Map a source category to Ontario category."""
    # Try exact match first
    if source_category in CATEGORY_MAPPINGS:
        return CATEGORY_MAPPINGS[source_category]
    
    # Try case-insensitive match
    source_lower = source_category.lower()
    for key, ontario_cat in CATEGORY_MAPPINGS.items():
        if key.lower() == source_lower:
            return ontario_cat
    
    # Try partial match
    for key, ontario_cat in CATEGORY_MAPPINGS.items():
        if key.lower() in source_lower or source_lower in key.lower():
            return ontario_cat
    
    # Default to garbage if unknown
    logger.warning(f"Unknown category '{source_category}', mapping to 'garbage'")
    return "garbage"


def copy_and_validate_image(src: Path, dst: Path) -> bool:
    """Copy image, validate and convert if necessary."""
    try:
        img = Image.open(src)
        # Convert to RGB if necessary
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        
        # Resize if too large (max 1024x1024)
        max_size = 1024
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Save as JPEG
        img.save(dst, 'JPEG', quality=95)
        return True
    except Exception as e:
        logger.warning(f"Failed to process {src}: {e}")
        return False


def split_dataset(
    images: List[Tuple[Path, str]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List, List, List]:
    """Split images into train/val/test sets."""
    random.seed(seed)
    images = list(images)
    random.shuffle(images)
    
    n = len(images)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train = images[:train_end]
    val = images[train_end:val_end]
    test = images[val_end:]
    
    return train, val, test


def prepare_dataset(
    data_dir: Path,
    output_dir: Path,
    min_images_per_category: int = 100,
) -> Dict:
    """Prepare the complete dataset."""
    logger.info("=" * 60)
    logger.info("PREPARING EXTENDED DATASET")
    logger.info("=" * 60)
    
    # Collect all images by Ontario category
    images_by_ontario: Dict[str, List[Tuple[Path, str]]] = {cat: [] for cat in ONTARIO_CATEGORIES.keys()}
    
    # Scan raw data directories
    raw_dir = data_dir / "raw"
    
    # Process TrashNet
    trashnet_dir = raw_dir / "trashnet" / "dataset-resized"
    if trashnet_dir.exists():
        logger.info("Processing TrashNet dataset...")
        for category_dir in trashnet_dir.iterdir():
            if category_dir.is_dir():
                ontario_cat = map_to_ontario_category(category_dir.name)
                for img_path in category_dir.iterdir():
                    if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                        images_by_ontario[ontario_cat].append((img_path, "trashnet"))
        logger.info(f"  Found {sum(len(v) for v in images_by_ontario.values())} images in TrashNet")
    
    # Process RealWaste
    realwaste_dir = raw_dir / "realwaste" / "realwaste-main" / "RealWaste"
    if realwaste_dir.exists():
        logger.info("Processing RealWaste dataset...")
        for category_dir in realwaste_dir.iterdir():
            if category_dir.is_dir():
                ontario_cat = map_to_ontario_category(category_dir.name)
                for img_path in category_dir.iterdir():
                    if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                        images_by_ontario[ontario_cat].append((img_path, "realwaste"))
        logger.info(f"  Found {sum(len(v) for v in images_by_ontario.values())} total images after RealWaste")
    
    # Process additional Kaggle datasets if present
    kaggle_dirs = [
        raw_dir / "garbage-classification" / "Garbage classification",
        raw_dir / "recyclable-waste",
        raw_dir / "waste-classification-data",
    ]
    
    for kaggle_dir in kaggle_dirs:
        if kaggle_dir.exists():
            logger.info(f"Processing {kaggle_dir.name}...")
            dataset_name = kaggle_dir.name
            for category_dir in kaggle_dir.iterdir():
                if category_dir.is_dir():
                    ontario_cat = map_to_ontario_category(category_dir.name)
                    for img_path in category_dir.iterdir():
                        if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                            images_by_ontario[ontario_cat].append((img_path, dataset_name))
    
    # Process E-Waste if present
    ewaste_dir = raw_dir / "e-waste"
    if ewaste_dir.exists():
        logger.info("Processing E-Waste dataset...")
        for img_path in ewaste_dir.rglob("*"):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                images_by_ontario["e_waste"].append((img_path, "e-waste"))
    
    # Process Hazardous if present
    hazardous_dir = raw_dir / "hazardous"
    if hazardous_dir.exists():
        logger.info("Processing Hazardous dataset...")
        for img_path in hazardous_dir.rglob("*"):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                images_by_ontario["hazardous"].append((img_path, "hazardous"))
    
    # Report category counts
    logger.info("\nImages per category:")
    stats = {}
    for cat, images in images_by_ontario.items():
        logger.info(f"  {ONTARIO_CATEGORIES[cat]}: {len(images)} images")
        stats[cat] = len(images)
    
    # Check for categories with insufficient data
    for cat, count in stats.items():
        if count < min_images_per_category:
            logger.warning(f"  {cat} has only {count} images (recommended: {min_images_per_category})")
            logger.warning(f"  Consider adding more {ONTARIO_CATEGORIES[cat]} images")
    
    # Create output directories
    logger.info("\nCreating dataset splits...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ["train", "val", "test"]:
        for cat in ONTARIO_CATEGORIES.keys():
            (output_dir / split / cat).mkdir(parents=True, exist_ok=True)
    
    # Split and copy images
    final_stats = {"train": {}, "val": {}, "test": {}}
    
    for ontario_cat, images in images_by_ontario.items():
        if not images:
            logger.warning(f"No images for {ontario_cat}, skipping...")
            continue
        
        train, val, test = split_dataset(images)
        
        # Copy train images
        for img_path, source in train:
            dst = output_dir / "train" / ontario_cat / f"{source}_{img_path.name}"
            if copy_and_validate_image(img_path, dst):
                final_stats["train"][ontario_cat] = final_stats["train"].get(ontario_cat, 0) + 1
        
        # Copy val images
        for img_path, source in val:
            dst = output_dir / "val" / ontario_cat / f"{source}_{img_path.name}"
            if copy_and_validate_image(img_path, dst):
                final_stats["val"][ontario_cat] = final_stats["val"].get(ontario_cat, 0) + 1
        
        # Copy test images
        for img_path, source in test:
            dst = output_dir / "test" / ontario_cat / f"{source}_{img_path.name}"
            if copy_and_validate_image(img_path, dst):
                final_stats["test"][ontario_cat] = final_stats["test"].get(ontario_cat, 0) + 1
    
    # Save stats
    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, 'w') as f:
        json.dump({
            "total_by_category": stats,
            "splits": final_stats,
            "categories": ONTARIO_CATEGORIES,
        }, f, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info("DATASET PREPARATION COMPLETE")
    logger.info("=" * 60)
    
    logger.info("\nFinal dataset statistics:")
    for split in ["train", "val", "test"]:
        total = sum(final_stats[split].values())
        logger.info(f"\n{split.upper()}: {total} images")
        for cat, count in sorted(final_stats[split].items()):
            logger.info(f"  {ONTARIO_CATEGORIES[cat]}: {count}")
    
    logger.info(f"\nDataset saved to: {output_dir}")
    logger.info(f"Stats saved to: {stats_path}")
    
    return final_stats


def main():
    parser = argparse.ArgumentParser(description="Prepare extended waste dataset")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="data/processed/ontario_extended", help="Output directory")
    parser.add_argument("--min-images", type=int, default=100, help="Minimum images per category")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    stats = prepare_dataset(data_dir, output_dir, args.min_images)
    
    # Print recommendations
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 60)
    
    min_images = min(stats["train"].values()) if stats["train"] else 0
    if min_images < 200:
        logger.warning("\nSome categories have limited training data.")
        logger.warning("For better accuracy, consider:")
        logger.warning("  1. Downloading additional datasets (see scripts/download_extended_datasets.py)")
        logger.warning("  2. Collecting more images for under-represented categories")
        logger.warning("  3. Using data augmentation during training")


if __name__ == "__main__":
    main()
